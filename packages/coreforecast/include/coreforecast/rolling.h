#pragma once

#include "SkipList.h"

#include "stats.h"

namespace rolling {
template <typename T> class MeanAccumulator {
public:
  MeanAccumulator(int window_size) : window_size_(window_size) {}
  void Update(T x) { accum_ += x; }
  T Update(T x, int n) {
    accum_ += x;
    return accum_ / static_cast<T>(n);
  }
  T Update(T new_x, T old_x) {
    accum_ += new_x - old_x;
    return accum_ / static_cast<T>(window_size_);
  }

private:
  int window_size_;
  T accum_ = 0.0;
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
                          int min_samples) {
  Transform<T, MeanAccumulator<T>>(data, n, out, window_size, min_samples);
}

template <typename T>
inline void StdTransformWithStats(const T *data, int n, T *out, T *agg,
                                  bool save_stats, int window_size,
                                  int min_samples) {
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
}

template <typename T>
inline void StdTransform(const T *data, int n, T *out, int window_size,
                         int min_samples) {
  T tmp;
  StdTransformWithStats(data, n, out, &tmp, false, window_size, min_samples);
}

template <typename T, typename Comp> class CompAccumulator {
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
    return Front().second;
  }
  T Update(T new_x, T old_x) noexcept {
    Insert(new_x);
    return Front().second;
  }

private:
  std::vector<std::pair<int, T>> buffer_;
  int window_size_;
  int head_ = 0;
  int tail_ = -1;
  int i_ = 0;
  Comp comp_ = Comp();
};

template <typename T>
void MinTransform(const T *data, int n, T *out, int window_size,
                  int min_samples) {
  Transform<T, CompAccumulator<T, std::greater_equal<T>>>(
      data, n, out, window_size, min_samples);
}

template <typename T>
void MaxTransform(const T *data, int n, T *out, int window_size,
                  int min_samples) {
  Transform<T, CompAccumulator<T, std::less_equal<T>>>(
      data, n, out, window_size, min_samples);
}

template <typename T> class QuantileAccumulator {
public:
  QuantileAccumulator(int window_size, T p)
      : window_size_(window_size), p_(p) {}
  void Update(T x) { skip_list_.insert(x); }
  T Update(T x, int n) {
    skip_list_.insert(x);
    return stats::SortedQuantile(skip_list_, p_, n);
  }
  T Update(T new_x, T old_x) {
    skip_list_.remove(old_x);
    skip_list_.insert(new_x);
    return stats::SortedQuantile(skip_list_, p_, window_size_);
  }

private:
  int window_size_;
  T p_;
  OrderedStructs::SkipList::HeadNode<T> skip_list_;
};

template <typename T>
inline void QuantileTransform(const T *data, int n, T *out, int window_size,
                              int min_samples, T p) {
  Transform<T, QuantileAccumulator<T>>(data, n, out, window_size, min_samples,
                                       p);
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
                                  int min_samples) {
  SeasonalTransform(MeanTransform<T>, data, n, out, season_length, window_size,
                    min_samples);
};

template <typename T>
inline void SeasonalStdTransform(const T *data, int n, T *out,
                                 int season_length, int window_size,
                                 int min_samples) {
  SeasonalTransform(StdTransform<T>, data, n, out, season_length, window_size,
                    min_samples);
};

template <typename T>
inline void SeasonalMinTransform(const T *data, int n, T *out,
                                 int season_length, int window_size,
                                 int min_samples) {
  SeasonalTransform(MinTransform<T>, data, n, out, season_length, window_size,
                    min_samples);
};

template <typename T>
inline void SeasonalMaxTransform(const T *data, int n, T *out,
                                 int season_length, int window_size,
                                 int min_samples) {
  SeasonalTransform(MaxTransform<T>, data, n, out, season_length, window_size,
                    min_samples);
};

template <typename T>
void SeasonalQuantileTransform(const T *data, int n, T *out, int season_length,
                               int window_size, int min_samples, T p) {
  SeasonalTransform(QuantileTransform<T>, data, n, out, season_length,
                    window_size, min_samples, p);
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
void MeanUpdate(const T *data, int n, T *out, int window_size,
                int min_samples) {
  Update(MeanTransform<T>, data, n, out, window_size, min_samples);
};

template <typename T>
void StdUpdate(const T *data, int n, T *out, int window_size, int min_samples) {
  Update(StdTransform<T>, data, n, out, window_size, min_samples);
};

template <typename T>
void MinUpdate(const T *data, int n, T *out, int window_size, int min_samples) {
  Update(MinTransform<T>, data, n, out, window_size, min_samples);
};

template <typename T>
void MaxUpdate(const T *data, int n, T *out, int window_size, int min_samples) {
  Update(MaxTransform<T>, data, n, out, window_size, min_samples);
};

template <typename T>
void QuantileUpdate(const T *data, int n, T *out, int window_size,
                    int min_samples, T p) {
  Update(QuantileTransform<T>, data, n, out, window_size, min_samples, p);
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
                        int window_size, int min_samples) {
  SeasonalUpdate(MeanUpdate<T>, data, n, out, season_length, window_size,
                 min_samples);
};

template <typename T>
void SeasonalStdUpdate(const T *data, int n, T *out, int season_length,
                       int window_size, int min_samples) {
  SeasonalUpdate(StdUpdate<T>, data, n, out, season_length, window_size,
                 min_samples);
};

template <typename T>
void SeasonalMinUpdate(const T *data, int n, T *out, int season_length,
                       int window_size, int min_samples) {
  SeasonalUpdate(MinUpdate<T>, data, n, out, season_length, window_size,
                 min_samples);
};

template <typename T>
void SeasonalMaxUpdate(const T *data, int n, T *out, int season_length,
                       int window_size, int min_samples) {
  SeasonalUpdate(MaxUpdate<T>, data, n, out, season_length, window_size,
                 min_samples);
};

template <typename T>
void SeasonalQuantileUpdate(const T *data, int n, T *out, int season_length,
                            int window_size, int min_samples, T p) {
  SeasonalUpdate(QuantileUpdate<T>, data, n, out, season_length, window_size,
                 min_samples, p);
};
} // namespace rolling
