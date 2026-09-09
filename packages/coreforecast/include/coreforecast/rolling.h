#pragma once

#include "SkipList.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
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

// ============================================================================
// CompAccumulator - PARTIALLY OPTIMIZED for SkipNA=false
// ============================================================================
// Cannot fully remove isnan checks because NaN breaks comparison semantics
// (NaN < x and NaN > x are both false), which corrupts the monotonic deque.
// Optimization: Short-circuit all deque operations once has_nan_ is true.
// ============================================================================
template <typename T, typename Comp, bool SkipNA> class CompAccumulator {
public:
  // std::pair's default constructor value-initializes, which would zero the
  // whole ring buffer on every allocation; this aggregate's one is trivial
  struct Entry {
    index_t index;
    T value;
  };

  CompAccumulator(index_t window_size)
      : buffer_(std::make_unique_for_overwrite<Entry[]>(
            static_cast<size_t>(window_size))),
        window_size_(window_size) {}
  inline bool Empty() const noexcept { return tail_ == -1; }
  inline void PushBack(index_t i, T x) noexcept {
    if (tail_ == -1) {
      head_ = 0;
      tail_ = 0;
    } else if (tail_ == window_size_ - 1) {
      tail_ = 0;
    } else {
      ++tail_;
    }
    buffer_[tail_] = {i, x};
  }
  inline void PopBack() noexcept {
    if (head_ == tail_) {
      head_ = 0;
      tail_ = -1;
    } else if (tail_ == 0) {
      tail_ = window_size_ - 1;
    } else {
      --tail_;
    }
  }
  inline void PopFront() noexcept {
    if (head_ == tail_) {
      head_ = 0;
      tail_ = -1;
    } else if (head_ == window_size_ - 1) {
      head_ = 0;
    } else {
      ++head_;
    }
  }
  inline const Entry &Front() const noexcept { return buffer_[head_]; }
  inline const Entry &Back() const noexcept { return buffer_[tail_]; }

  void Insert(T x) noexcept {
    if constexpr (!SkipNA) {
      // Short-circuit: once NaN seen, skip all deque maintenance
      if (has_nan_) {
        ++i_;
        return;
      }
      if (std::isnan(x)) {
        has_nan_ = true;
        ++i_;
        return;
      }
    }

    if constexpr (SkipNA) {
      if (std::isnan(x)) {
        // the front can expire on this step even though nothing is inserted;
        // skipping the check leaves an out-of-window entry to be returned
        if (!Empty() && Front().index <= i_) {
          PopFront();
        }
        ++i_;
        return;
      }
    }

    // Valid value: maintain monotonic deque
    while (!Empty() && comp_(Back().value, x)) {
      PopBack();
    }
    if (!Empty() && Front().index <= i_) {
      PopFront();
    }
    PushBack(window_size_ + i_, x);
    ++i_;
  }

  void Update(T x) noexcept { Insert(x); }

  T Update(T x, index_t) noexcept {
    Insert(x);
    if constexpr (!SkipNA) {
      if (has_nan_)
        return kNaN<T>;
    }
    if (Empty())
      return kNaN<T>;
    return Front().value;
  }

  T Update(T new_x, T) noexcept {
    Insert(new_x);
    if constexpr (!SkipNA) {
      if (has_nan_)
        return kNaN<T>;
    }
    if (Empty())
      return kNaN<T>;
    return Front().value;
  }

private:
  std::unique_ptr<Entry[]> buffer_;
  index_t window_size_;
  index_t head_ = 0;
  index_t tail_ = -1;
  index_t i_ = 0;
  [[no_unique_address]] std::conditional_t<SkipNA, std::monostate, bool>
      has_nan_{};
  Comp comp_ = Comp();
};

template <typename T>
void MinTransform(std::span<const T> data, std::span<T> out, const Window &w) {
  if (w.skipna) {
    Transform<T, CompAccumulator<T, std::greater_equal<T>, true>>(data, out, w);
  } else {
    Transform<T, CompAccumulator<T, std::greater_equal<T>, false>>(data, out,
                                                                   w);
  }
}

template <typename T>
void MaxTransform(std::span<const T> data, std::span<T> out, const Window &w) {
  if (w.skipna) {
    Transform<T, CompAccumulator<T, std::less_equal<T>, true>>(data, out, w);
  } else {
    Transform<T, CompAccumulator<T, std::less_equal<T>, false>>(data, out, w);
  }
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
