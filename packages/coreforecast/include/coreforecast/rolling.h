#pragma once

#include "SkipList.h"

#include "stats.h"
#include <variant>

namespace rolling {

template <typename T, bool SkipNA> class MeanAccumulator {
public:
  MeanAccumulator(int window_size) : window_size_(window_size) {}

  void Update(T x) {
    if constexpr (SkipNA) {
      if (std::isnan(x))
        return;
      valid_count_++;
    }
    accum_ += x;
  }

  T Update(T x, int n) {
    Update(x);
    if constexpr (SkipNA) {
      if (valid_count_ == 0)
        return std::numeric_limits<T>::quiet_NaN();
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
        return std::numeric_limits<T>::quiet_NaN();
      return accum_ / static_cast<T>(valid_count_);
    } else {
      // No isnan check - NaN propagates naturally through arithmetic
      accum_ += new_x - old_x;
      return accum_ / static_cast<T>(window_size_);
    }
  }

private:
  int window_size_;
  T accum_ = 0.0;
  // valid_count_ only exists when SkipNA=true
  [[no_unique_address]] std::conditional_t<SkipNA, int, std::monostate>
      valid_count_{};
};

template <typename T, typename Accumulator, typename... Args>
inline void Transform(const T *data, int n, T *out, int window_size,
                      int min_samples, Args &&...args) {
  if (n < min_samples) {
    std::fill(out, out + n, std::numeric_limits<T>::quiet_NaN());
    return;
  }
  Accumulator accumulator(window_size, std::forward<Args>(args)...);
  window_size = std::min(window_size, n);
  min_samples = std::min(min_samples, window_size);
  for (int i = 0; i < min_samples - 1; ++i) {
    accumulator.Update(data[i]);
    out[i] = std::numeric_limits<T>::quiet_NaN();
  }
  for (int i = min_samples - 1; i < window_size; ++i) {
    out[i] = accumulator.Update(data[i], i + 1);
  }
  for (int i = window_size; i < n; ++i) {
    out[i] = accumulator.Update(data[i], data[i - window_size]);
  }
}

template <typename T>
inline void MeanTransform(const T *data, int n, T *out, int window_size,
                          int min_samples, bool skipna = false) {
  if (skipna) {
    Transform<T, MeanAccumulator<T, true>>(data, n, out, window_size,
                                           min_samples);
  } else {
    Transform<T, MeanAccumulator<T, false>>(data, n, out, window_size,
                                            min_samples);
  }
}

template <typename T>
inline void StdTransformWithStats(const T *data, int n, T *out, T *agg,
                                  bool save_stats, int window_size,
                                  int min_samples, bool skipna = false) {
  if (!skipna) {
    // Fast path: original implementation without NaN checking
    T prev_avg = static_cast<T>(0.0);
    T curr_avg = data[0];
    T m2 = static_cast<T>(0.0);
    int upper_limit = std::min(window_size, n);
    for (int i = 0; i < upper_limit; ++i) {
      prev_avg = curr_avg;
      curr_avg = prev_avg + (data[i] - prev_avg) / (i + 1);
      m2 += (data[i] - prev_avg) * (data[i] - curr_avg);
      if (i + 1 < min_samples) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
      } else {
        out[i] = sqrt(m2 / i);
      }
    }
    for (int i = window_size; i < n; ++i) {
      T delta = data[i] - data[i - window_size];
      prev_avg = curr_avg;
      curr_avg = prev_avg + delta / window_size;
      m2 += delta * (data[i] - curr_avg + data[i - window_size] - prev_avg);
      // avoid possible loss of precision
      m2 = std::max(m2, static_cast<T>(0.0));
      out[i] = sqrt(m2 / (window_size - 1));
    }
    if (save_stats) {
      agg[0] = static_cast<T>(n);
      agg[1] = curr_avg;
      agg[2] = m2;
    }
  } else {
    // Slow path: NaN-aware Welford's algorithm
    T mean = 0.0;
    T m2 = 0.0;
    int count = 0;
    int upper_limit = std::min(window_size, n);
    // Expanding window phase
    for (int i = 0; i < upper_limit; ++i) {
      if (!std::isnan(data[i])) {
        count++;
        T delta = data[i] - mean;
        mean += delta / count;
        T delta2 = data[i] - mean;
        m2 += delta * delta2;
      }
      if (count < min_samples) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
      } else if (count <= 1) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
      } else {
        T var = m2 / (count - 1);
        out[i] = sqrt(var < 0 ? 0.0 : var);
      }
    }
    // Sliding window phase
    for (int i = window_size; i < n; ++i) {
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
        out[i] = std::numeric_limits<T>::quiet_NaN();
      } else if (count <= 1) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
      } else {
        T var = m2 / (count - 1);
        out[i] = sqrt(var < 0 ? 0.0 : var);
      }
    }
    if (save_stats) {
      agg[0] = static_cast<T>(count);
      agg[1] = mean;
      agg[2] = m2;
    }
  }
}

template <typename T>
inline void StdTransform(const T *data, int n, T *out, int window_size,
                         int min_samples, bool skipna = false) {
  T tmp;
  StdTransformWithStats(data, n, out, &tmp, false, window_size, min_samples,
                        skipna);
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
  CompAccumulator(int window_size) : window_size_(window_size) {
    buffer_.reserve(window_size);
  }
  inline bool Empty() const noexcept { return tail_ == -1; }
  inline void PushBack(int i, T x) noexcept {
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
  inline const std::pair<int, T> &Front() const noexcept {
    return buffer_[head_];
  }
  inline const std::pair<int, T> &Back() const noexcept {
    return buffer_[tail_];
  }

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
        ++i_;
        return;
      }
    }

    // Valid value: maintain monotonic deque
    while (!Empty() && comp_(Back().second, x)) {
      PopBack();
    }
    if (!Empty() && Front().first <= i_) {
      PopFront();
    }
    PushBack(window_size_ + i_, x);
    ++i_;
  }

  void Update(T x) noexcept { Insert(x); }

  T Update(T x, int n) noexcept {
    Insert(x);
    if constexpr (!SkipNA) {
      if (has_nan_)
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (Empty())
      return std::numeric_limits<T>::quiet_NaN();
    return Front().second;
  }

  T Update(T new_x, T old_x) noexcept {
    Insert(new_x);
    if constexpr (!SkipNA) {
      if (has_nan_)
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (Empty())
      return std::numeric_limits<T>::quiet_NaN();
    return Front().second;
  }

private:
  std::vector<std::pair<int, T>> buffer_;
  int window_size_;
  int head_ = 0;
  int tail_ = -1;
  int i_ = 0;
  [[no_unique_address]] std::conditional_t<SkipNA, std::monostate, bool>
      has_nan_{};
  Comp comp_ = Comp();
};

template <typename T>
void MinTransform(const T *data, int n, T *out, int window_size,
                  int min_samples, bool skipna = false) {
  if (skipna) {
    Transform<T, CompAccumulator<T, std::greater_equal<T>, true>>(
        data, n, out, window_size, min_samples);
  } else {
    Transform<T, CompAccumulator<T, std::greater_equal<T>, false>>(
        data, n, out, window_size, min_samples);
  }
}

template <typename T>
void MaxTransform(const T *data, int n, T *out, int window_size,
                  int min_samples, bool skipna = false) {
  if (skipna) {
    Transform<T, CompAccumulator<T, std::less_equal<T>, true>>(
        data, n, out, window_size, min_samples);
  } else {
    Transform<T, CompAccumulator<T, std::less_equal<T>, false>>(
        data, n, out, window_size, min_samples);
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
  QuantileAccumulator(int window_size, T p)
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

  T Update(T x, int n) {
    Update(x);
    if constexpr (!SkipNA) {
      if (has_nan_)
        return std::numeric_limits<T>::quiet_NaN();
      return stats::SortedQuantile(skip_list_, p_, n);
    } else {
      if (valid_count_ == 0)
        return std::numeric_limits<T>::quiet_NaN();
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
        return std::numeric_limits<T>::quiet_NaN();
      return stats::SortedQuantile(skip_list_, p_, valid_count_);
    } else {
      // Handle NaN entering or leaving window
      bool old_is_nan = std::isnan(old_x);
      bool new_is_nan = std::isnan(new_x);

      if (new_is_nan) {
        has_nan_ = true;
      }

      if (has_nan_)
        return std::numeric_limits<T>::quiet_NaN();

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
  int window_size_;
  T p_;
  [[no_unique_address]] std::conditional_t<SkipNA, int, std::monostate>
      valid_count_{};
  [[no_unique_address]] std::conditional_t<SkipNA, std::monostate, bool>
      has_nan_{};
  OrderedStructs::SkipList::HeadNode<T> skip_list_;
};

template <typename T>
inline void QuantileTransform(const T *data, int n, T *out, int window_size,
                              int min_samples, T p, bool skipna = false) {
  if (skipna) {
    Transform<T, QuantileAccumulator<T, true>>(data, n, out, window_size,
                                               min_samples, p);
  } else {
    Transform<T, QuantileAccumulator<T, false>>(data, n, out, window_size,
                                                min_samples, p);
  }
}

template <typename Func, typename T, typename... Args>
inline void SeasonalTransform(Func RollingTfm, const T *data, int n, T *out,
                              int season_length, int window_size,
                              int min_samples, Args &&...args) {
  int buff_size = n / season_length + (n % season_length > 0);
  std::vector<T> season_data(buff_size);
  std::vector<T> season_out(buff_size);
  for (int i = 0; i < std::min(n, season_length); ++i) {
    int season_n = n / season_length + (i < n % season_length);
    for (int j = 0; j < season_n; ++j) {
      season_data[j] = data[i + j * season_length];
    }
    RollingTfm(season_data.data(), season_n, season_out.data(), window_size,
               min_samples, std::forward<Args>(args)...);
    for (int j = 0; j < season_n; ++j) {
      out[i + j * season_length] = season_out[j];
    }
  }
}

template <typename T>
inline void SeasonalMeanTransform(const T *data, int n, T *out,
                                  int season_length, int window_size,
                                  int min_samples, bool skipna = false) {
  SeasonalTransform(MeanTransform<T>, data, n, out, season_length, window_size,
                    min_samples, skipna);
};

template <typename T>
inline void SeasonalStdTransform(const T *data, int n, T *out,
                                 int season_length, int window_size,
                                 int min_samples, bool skipna = false) {
  SeasonalTransform(StdTransform<T>, data, n, out, season_length, window_size,
                    min_samples, skipna);
};

template <typename T>
inline void SeasonalMinTransform(const T *data, int n, T *out,
                                 int season_length, int window_size,
                                 int min_samples, bool skipna = false) {
  SeasonalTransform(MinTransform<T>, data, n, out, season_length, window_size,
                    min_samples, skipna);
};

template <typename T>
inline void SeasonalMaxTransform(const T *data, int n, T *out,
                                 int season_length, int window_size,
                                 int min_samples, bool skipna = false) {
  SeasonalTransform(MaxTransform<T>, data, n, out, season_length, window_size,
                    min_samples, skipna);
};

template <typename T>
void SeasonalQuantileTransform(const T *data, int n, T *out, int season_length,
                               int window_size, int min_samples, T p,
                               bool skipna = false) {
  SeasonalTransform(QuantileTransform<T>, data, n, out, season_length,
                    window_size, min_samples, p, skipna);
};

template <typename Func, typename T, typename... Args>
inline void Update(Func RollingTfm, const T *data, int n, T *out,
                   int window_size, int min_samples, Args &&...args) {
  if (n < min_samples) {
    *out = std::numeric_limits<T>::quiet_NaN();
    return;
  }
  int n_samples = std::min(window_size, n);
  std::vector<T> buffer(n_samples);
  RollingTfm(data + n - n_samples, n_samples, buffer.data(), window_size,
             min_samples, std::forward<Args>(args)...);
  *out = buffer[n_samples - 1];
}

template <typename T>
void MeanUpdate(const T *data, int n, T *out, int window_size, int min_samples,
                bool skipna = false) {
  Update(MeanTransform<T>, data, n, out, window_size, min_samples, skipna);
};

template <typename T>
void StdUpdate(const T *data, int n, T *out, int window_size, int min_samples,
               bool skipna = false) {
  Update(StdTransform<T>, data, n, out, window_size, min_samples, skipna);
};

template <typename T>
void MinUpdate(const T *data, int n, T *out, int window_size, int min_samples,
               bool skipna = false) {
  Update(MinTransform<T>, data, n, out, window_size, min_samples, skipna);
};

template <typename T>
void MaxUpdate(const T *data, int n, T *out, int window_size, int min_samples,
               bool skipna = false) {
  Update(MaxTransform<T>, data, n, out, window_size, min_samples, skipna);
};

template <typename T>
void QuantileUpdate(const T *data, int n, T *out, int window_size,
                    int min_samples, T p, bool skipna = false) {
  Update(QuantileTransform<T>, data, n, out, window_size, min_samples, p,
         skipna);
};

template <typename Func, typename T, typename... Args>
inline void SeasonalUpdate(Func RollingUpdate, const T *data, int n, T *out,
                           int season_length, int window_size, int min_samples,
                           Args &&...args) {
  int season = n % season_length;
  int season_n = n / season_length + (season > 0);
  if (season_n < min_samples) {
    *out = std::numeric_limits<T>::quiet_NaN();
    return;
  }
  int n_samples = std::min(window_size, season_n);
  std::vector<T> season_data(n_samples);
  for (int i = 0; i < n_samples; ++i) {
    season_data[i] = data[n - 1 - (n_samples - 1 - i) * season_length];
  }
  RollingUpdate(season_data.data(), n_samples, out, window_size, min_samples,
                std::forward<Args>(args)...);
}

template <typename T>
void SeasonalMeanUpdate(const T *data, int n, T *out, int season_length,
                        int window_size, int min_samples, bool skipna = false) {
  SeasonalUpdate(MeanUpdate<T>, data, n, out, season_length, window_size,
                 min_samples, skipna);
};

template <typename T>
void SeasonalStdUpdate(const T *data, int n, T *out, int season_length,
                       int window_size, int min_samples, bool skipna = false) {
  SeasonalUpdate(StdUpdate<T>, data, n, out, season_length, window_size,
                 min_samples, skipna);
};

template <typename T>
void SeasonalMinUpdate(const T *data, int n, T *out, int season_length,
                       int window_size, int min_samples, bool skipna = false) {
  SeasonalUpdate(MinUpdate<T>, data, n, out, season_length, window_size,
                 min_samples, skipna);
};

template <typename T>
void SeasonalMaxUpdate(const T *data, int n, T *out, int season_length,
                       int window_size, int min_samples, bool skipna = false) {
  SeasonalUpdate(MaxUpdate<T>, data, n, out, season_length, window_size,
                 min_samples, skipna);
};

template <typename T>
void SeasonalQuantileUpdate(const T *data, int n, T *out, int season_length,
                            int window_size, int min_samples, T p,
                            bool skipna = false) {
  SeasonalUpdate(QuantileUpdate<T>, data, n, out, season_length, window_size,
                 min_samples, p, skipna);
};
} // namespace rolling