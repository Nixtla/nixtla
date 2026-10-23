#pragma once

#include "SkipList.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <span>
#include <type_traits>
#include <variant>
#include <vector>

#include "common.h"
#include "stats.h"

namespace rolling {

template <typename T, bool SkipNA> class MeanAccumulator {
public:
  MeanAccumulator(index_t window_size) : window_size_(window_size) {}

  void Update(T x) {
    if constexpr (SkipNA) {
      if (std::isnan(x))
        return;
      valid_count_++;
    }
    accum_ += x;
  }

  T Update(T x, index_t n) {
    Update(x);
    if constexpr (SkipNA) {
      if (valid_count_ == 0)
        return kNaN<T>;
      return accum_ / static_cast<T>(valid_count_);
    } else {
      // NaN in accum_ naturally produces NaN result
      return accum_ / static_cast<T>(n);
    }
  }

  T Update(T new_x, T old_x) {
    if constexpr (SkipNA) {
      if (!std::isnan(old_x)) {
        accum_ -= old_x;
        valid_count_--;
      }
      if (!std::isnan(new_x)) {
        accum_ += new_x;
        valid_count_++;
      }
      if (valid_count_ == 0)
        return kNaN<T>;
      return accum_ / static_cast<T>(valid_count_);
    } else {
      // No isnan check - NaN propagates naturally through arithmetic
      accum_ += new_x - old_x;
      return accum_ / static_cast<T>(window_size_);
    }
  }

private:
  index_t window_size_;
  T accum_ = 0.0;
  // valid_count_ only exists when SkipNA=true
  [[no_unique_address]] std::conditional_t<SkipNA, index_t, std::monostate>
      valid_count_{};
};

template <typename T, typename Accumulator, typename... Args>
inline void Transform(std::span<const T> data, std::span<T> out,
                      const Window &w, Args &&...args) {
  assert(w.window_size > 0 && w.min_samples > 0);
  const index_t n = std::ssize(data);
  if (n < w.min_samples) {
    FillNaN(out);
    return;
  }
  Accumulator accumulator(w.window_size, std::forward<Args>(args)...);
  const index_t window_size = std::min(w.window_size, n);
  const index_t min_samples = std::min(w.min_samples, window_size);
  for (index_t i = 0; i < min_samples - 1; ++i) {
    accumulator.Update(data[i]);
    out[i] = kNaN<T>;
  }
  for (index_t i = min_samples - 1; i < window_size; ++i) {
    out[i] = accumulator.Update(data[i], i + 1);
  }
  for (index_t i = window_size; i < n; ++i) {
    out[i] = accumulator.Update(data[i], data[i - window_size]);
  }
}

template <typename T>
inline void MeanTransform(std::span<const T> data, std::span<T> out,
                          const Window &w) {
  if (w.skipna) {
    Transform<T, MeanAccumulator<T, true>>(data, out, w);
  } else {
    Transform<T, MeanAccumulator<T, false>>(data, out, w);
  }
}

// `agg` is either empty or three elements: the count, mean and M2 of the last
// window, which the expanding std uses to resume from.
template <typename T>
inline void StdTransformWithStats(std::span<const T> data, std::span<T> out,
                                  std::span<T> agg, const Window &w) {
  assert(w.window_size > 0 && w.min_samples > 0);
  const index_t n = std::ssize(data);
  const index_t window_size = w.window_size;
  const index_t min_samples = w.min_samples;
  if (!w.skipna) {
    // Fast path: original implementation without NaN checking
    T prev_avg = static_cast<T>(0.0);
    T curr_avg = data[0];
    T m2 = static_cast<T>(0.0);
    const index_t upper_limit = std::min(window_size, n);
    for (index_t i = 0; i < upper_limit; ++i) {
      prev_avg = curr_avg;
      curr_avg = prev_avg + (data[i] - prev_avg) / (i + 1);
      m2 += (data[i] - prev_avg) * (data[i] - curr_avg);
      if (i + 1 < min_samples) {
        out[i] = kNaN<T>;
      } else {
        out[i] = sqrt(m2 / i);
      }
    }
    for (index_t i = window_size; i < n; ++i) {
      T delta = data[i] - data[i - window_size];
      prev_avg = curr_avg;
      curr_avg = prev_avg + delta / window_size;
      m2 += delta * (data[i] - curr_avg + data[i - window_size] - prev_avg);
      // avoid possible loss of precision
      m2 = std::max(m2, static_cast<T>(0.0));
      out[i] = sqrt(m2 / (window_size - 1));
    }
    if (!agg.empty()) {
      agg[0] = static_cast<T>(n);
      agg[1] = curr_avg;
      agg[2] = m2;
    }
  } else {
    // Slow path: NaN-aware Welford's algorithm
    T mean = 0.0;
    T m2 = 0.0;
    index_t count = 0;
    const index_t upper_limit = std::min(window_size, n);
    // Expanding window phase
    for (index_t i = 0; i < upper_limit; ++i) {
      if (!std::isnan(data[i])) {
        count++;
        T delta = data[i] - mean;
        mean += delta / count;
        T delta2 = data[i] - mean;
        m2 += delta * delta2;
      }
      if (count < min_samples) {
        out[i] = kNaN<T>;
      } else if (count <= 1) {
        out[i] = kNaN<T>;
      } else {
        T var = m2 / (count - 1);
        out[i] = sqrt(var < 0 ? 0.0 : var);
      }
    }
    // Sliding window phase
    for (index_t i = window_size; i < n; ++i) {
      // Remove old value
      T old_val = data[i - window_size];
      if (!std::isnan(old_val)) {
        if (count == 1) {
          mean = m2 = 0.0;
          count = 0;
        } else {
          T delta = old_val - mean;
          mean -= delta / (count - 1);
          T delta2 = old_val - mean;
          m2 -= delta * delta2;
          count--;
        }
      }
      // Add new value
      if (!std::isnan(data[i])) {
        count++;
        T delta = data[i] - mean;
        mean += delta / count;
        T delta2 = data[i] - mean;
        m2 += delta * delta2;
      }
      if (count < min_samples) {
        out[i] = kNaN<T>;
      } else if (count <= 1) {
        out[i] = kNaN<T>;
      } else {
        T var = m2 / (count - 1);
        out[i] = sqrt(var < 0 ? 0.0 : var);
      }
    }
    if (!agg.empty()) {
      agg[0] = static_cast<T>(count);
      agg[1] = mean;
      agg[2] = m2;
    }
  }
}

template <typename T>
inline void StdTransform(std::span<const T> data, std::span<T> out,
                         const Window &w) {
  StdTransformWithStats(data, out, std::span<T>{}, w);
}

// Rolling min and max are one van Herk / Gil-Werman block scan: each output
// joins the suffix scan of the previous block with the prefix scan of the
// current one, three compare-selects per element and no data-dependent
// branch. The op must stay the plain `a < b ? b : a` (one maxsd on x86-64,
// fcsel on aarch64): std::fmax is a libm call on x86-64 GCC and an isnan()
// select becomes a branch. NaN handling therefore lives outside the op. With
// skipna a NaN turns into the identity before the op and a rolling NaN count
// makes a window without a valid value NaN, which the identity alone can't
// since -inf is a legitimate value. Without skipna the caller passes NaN-free
// data.
template <typename T> struct MaxOp {
  static constexpr T id = -std::numeric_limits<T>::infinity();
  T operator()(T a, T b) const noexcept { return a < b ? b : a; }
};

template <typename T> struct MinOp {
  static constexpr T id = std::numeric_limits<T>::infinity();
  T operator()(T a, T b) const noexcept { return b < a ? b : a; }
};

template <typename T, typename Op, bool SkipNA>
inline void BlockScan(std::span<const T> data, std::span<T> out, index_t w,
                      index_t min_samples) {
  const Op op;
  const index_t n = std::ssize(data);
  auto clean = [](T x) -> T {
    if constexpr (SkipNA)
      return std::isnan(x) ? Op::id : x;
    else
      return x;
  };
  auto is_nan = [](T x) -> index_t {
    if constexpr (SkipNA)
      return std::isnan(x);
    else
      return 0;
  };
  index_t nans = 0;
  T run = Op::id;
  const index_t head = std::min(w, n);
  for (index_t i = 0; i < head; ++i) {
    run = op(run, clean(data[i]));
    nans += is_nan(data[i]);
    out[i] = (i + 1 < min_samples || nans == i + 1) ? kNaN<T> : run;
  }
  if (n <= w)
    return;
  // slot w holds the identity so the window that is exactly one block needs
  // no special case
  std::vector<T> prev(w + 1, Op::id), cur(w + 1, Op::id);
  auto suffix = [&](std::vector<T> &buf, index_t start) {
    buf[w - 1] = clean(data[start + w - 1]);
    for (index_t j = w - 2; j >= 0; --j)
      buf[j] = op(clean(data[start + j]), buf[j + 1]);
  };
  suffix(prev, 0);
  for (index_t b = w; b < n; b += w) {
    const index_t len = std::min(w, n - b);
    run = Op::id;
    for (index_t o = 0; o < len; ++o) {
      const index_t i = b + o;
      run = op(run, clean(data[i]));
      nans += is_nan(data[i]) - is_nan(data[i - w]);
      out[i] = nans == w ? kNaN<T> : op(prev[o + 1], run);
    }
    if (b + len < n) {
      suffix(cur, b);
      std::swap(prev, cur);
    }
  }
}

template <typename T, typename Op>
inline void CompTransform(std::span<const T> data, std::span<T> out,
                          const Window &w) {
  assert(w.window_size > 0 && w.min_samples > 0);
  const index_t n = std::ssize(data);
  if (n < w.min_samples) {
    FillNaN(out);
    return;
  }
  if (w.skipna) {
    const index_t window_size = std::min(w.window_size, n);
    BlockScan<T, Op, true>(data, out, window_size,
                           std::min(w.min_samples, window_size));
    return;
  }
  // a NaN poisons its own output and every later one, so the kernel only
  // sees the clean prefix; this branch fires once
  index_t k = 0;
  while (k < n && !std::isnan(data[k]))
    ++k;
  if (k < w.min_samples) {
    FillNaN(out);
    return;
  }
  const index_t window_size = std::min(w.window_size, k);
  BlockScan<T, Op, false>(data.first(k), out.first(k), window_size,
                          std::min(w.min_samples, window_size));
  FillNaN(out.subspan(k));
}

template <typename T>
inline void MinTransform(std::span<const T> data, std::span<T> out,
                         const Window &w) {
  CompTransform<T, MinOp<T>>(data, out, w);
}

template <typename T>
inline void MaxTransform(std::span<const T> data, std::span<T> out,
                         const Window &w) {
  CompTransform<T, MaxOp<T>>(data, out, w);
}

// ============================================================================
// QuantileAccumulator - PARTIALLY OPTIMIZED for SkipNA=false
// ============================================================================
// Cannot fully remove isnan checks because NaN breaks skip list comparisons.
// Optimization: Don't insert NaN into skip list, just track the flag.
// ============================================================================
template <typename T, bool SkipNA> class QuantileAccumulator {
public:
  QuantileAccumulator(index_t window_size, T p)
      : window_size_(window_size), p_(p) {}

  void Update(T x) {
    if constexpr (SkipNA) {
      if (std::isnan(x))
        return;
      skip_list_.insert(x);
      valid_count_++;
    } else {
      if (std::isnan(x)) {
        has_nan_ = true;
        return; // Don't insert NaN into skip list
      }
      skip_list_.insert(x);
    }
  }

  T Update(T x, index_t n) {
    Update(x);
    if constexpr (!SkipNA) {
      if (has_nan_)
        return kNaN<T>;
      return stats::SortedQuantile(skip_list_, p_, n);
    } else {
      if (valid_count_ == 0)
        return kNaN<T>;
      return stats::SortedQuantile(skip_list_, p_, valid_count_);
    }
  }

  T Update(T new_x, T old_x) {
    if constexpr (SkipNA) {
      if (!std::isnan(old_x)) {
        skip_list_.remove(old_x);
        valid_count_--;
      }
      if (!std::isnan(new_x)) {
        skip_list_.insert(new_x);
        valid_count_++;
      }
      if (valid_count_ == 0)
        return kNaN<T>;
      return stats::SortedQuantile(skip_list_, p_, valid_count_);
    } else {
      // Handle NaN entering or leaving window
      bool old_is_nan = std::isnan(old_x);
      bool new_is_nan = std::isnan(new_x);

      if (new_is_nan) {
        has_nan_ = true;
      }

      if (has_nan_)
        return kNaN<T>;

      // Both values are valid (and no NaN seen yet)
      if (!old_is_nan) {
        skip_list_.remove(old_x);
      }
      if (!new_is_nan) {
        skip_list_.insert(new_x);
      }
      return stats::SortedQuantile(skip_list_, p_, window_size_);
    }
  }

private:
  index_t window_size_;
  T p_;
  [[no_unique_address]] std::conditional_t<SkipNA, index_t, std::monostate>
      valid_count_{};
  [[no_unique_address]] std::conditional_t<SkipNA, std::monostate, bool>
      has_nan_{};
  OrderedStructs::SkipList::HeadNode<T> skip_list_;
};

template <typename T>
inline void QuantileTransform(std::span<const T> data, std::span<T> out,
                              const Window &w, T p) {
  assert(p >= 0 && p <= 1);
  if (w.skipna) {
    Transform<T, QuantileAccumulator<T, true>>(data, out, w, p);
  } else {
    Transform<T, QuantileAccumulator<T, false>>(data, out, w, p);
  }
}

// Applies a rolling transform to each of the season_length phases of `data`.
template <typename Func, typename T, typename... Args>
inline void SeasonalTransform(Func RollingTfm, std::span<const T> data,
                              std::span<T> out, const SeasonalWindow &sw,
                              Args &&...args) {
  assert(sw.season_length > 0);
  const index_t season_length = sw.season_length;
  const index_t n = std::ssize(data);
  const index_t buff_size = n / season_length + (n % season_length > 0);
  std::vector<T> season_data(buff_size);
  std::vector<T> season_out(buff_size);
  for (index_t i = 0; i < std::min(n, season_length); ++i) {
    const index_t season_n = n / season_length + (i < n % season_length);
    for (index_t j = 0; j < season_n; ++j) {
      season_data[j] = data[i + j * season_length];
    }
    RollingTfm(std::span<const T>{season_data}.first(season_n),
               std::span<T>{season_out}.first(season_n), sw.window,
               std::forward<Args>(args)...);
    for (index_t j = 0; j < season_n; ++j) {
      out[i + j * season_length] = season_out[j];
    }
  }
}

// The value the rolling transform would put at the last position; out has
// one element.
template <typename Func, typename T, typename... Args>
inline void Update(Func RollingTfm, std::span<const T> data, std::span<T> out,
                   const Window &w, Args &&...args) {
  assert(w.window_size > 0);
  const index_t n = std::ssize(data);
  if (n < w.min_samples) {
    out[0] = kNaN<T>;
    return;
  }
  const index_t n_samples = std::min(w.window_size, n);
  std::vector<T> buffer(n_samples);
  RollingTfm(data.last(n_samples), std::span<T>{buffer}, w,
             std::forward<Args>(args)...);
  out[0] = buffer[n_samples - 1];
}

template <typename Func, typename T, typename... Args>
inline void SeasonalUpdate(Func RollingUpdate, std::span<const T> data,
                           std::span<T> out, const SeasonalWindow &sw,
                           Args &&...args) {
  assert(sw.season_length > 0 && sw.window.window_size > 0);
  const index_t season_length = sw.season_length;
  const index_t n = std::ssize(data);
  const index_t season = n % season_length;
  const index_t season_n = n / season_length + (season > 0);
  if (season_n < sw.window.min_samples) {
    out[0] = kNaN<T>;
    return;
  }
  const index_t n_samples = std::min(sw.window.window_size, season_n);
  std::vector<T> season_data(n_samples);
  for (index_t i = 0; i < n_samples; ++i) {
    season_data[i] = data[n - 1 - (n_samples - 1 - i) * season_length];
  }
  RollingUpdate(std::span<const T>{season_data}, out, sw.window,
                std::forward<Args>(args)...);
}

} // namespace rolling
