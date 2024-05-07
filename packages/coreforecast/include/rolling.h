#pragma once

#include "SkipList.h"

#include "grouped_array.h"
#include "stats.h"

template <typename T>
inline void RollingMeanTransform(const T *data, int n, T *out, int window_size,
                                 int min_samples) {
  T accum = static_cast<T>(0.0);
  int upper_limit = std::min(window_size, n);
  for (int i = 0; i < upper_limit; ++i) {
    accum += data[i];
    if (i + 1 < min_samples) {
      out[i] = std::numeric_limits<T>::quiet_NaN();
    } else {
      out[i] = accum / (i + 1);
    }
  }

  for (int i = window_size; i < n; ++i) {
    accum += data[i] - data[i - window_size];
    out[i] = accum / window_size;
  }
}

template <typename T>
inline void RollingStdTransformWithStats(const T *data, int n, T *out, T *agg,
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
inline void RollingStdTransform(const T *data, int n, T *out, int window_size,
                                int min_samples) {
  T tmp;
  RollingStdTransformWithStats(data, n, out, &tmp, false, window_size,
                               min_samples);
}

template <typename T, typename Comp> class SortedDeque {
public:
  SortedDeque(int window_size, Comp comp = Comp())
      : window_size_(window_size), comp_(comp) {
    buffer_.reserve(window_size);
  }
  inline bool empty() const noexcept { return tail_ == -1; }
  inline void push_back(int i, T x) noexcept {
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
  inline void pop_back() noexcept {
    if (head_ == tail_) {
      head_ = 0;
      tail_ = -1;
    } else if (tail_ == 0) {
      tail_ = window_size_ - 1;
    } else {
      --tail_;
    }
  }
  inline void pop_front() noexcept {
    if (head_ == tail_) {
      head_ = 0;
      tail_ = -1;
    } else if (head_ == window_size_ - 1) {
      head_ = 0;
    } else {
      ++head_;
    }
  }
  inline const std::pair<int, T> &front() const noexcept {
    return buffer_[head_];
  }
  inline const std::pair<int, T> &back() const noexcept {
    return buffer_[tail_];
  }
  void update(T x) noexcept {
    while (!empty() && comp_(back().second, x)) {
      pop_back();
    }
    if (!empty() && front().first <= i_) {
      pop_front();
    }
    push_back(window_size_ + i_, x);
    ++i_;
  }
  T get() const noexcept { return front().second; }

private:
  std::vector<std::pair<int, T>> buffer_;
  int window_size_;
  int head_ = 0;
  int tail_ = -1;
  int i_ = 0;
  Comp comp_;
};

template <typename T, typename Comp>
inline void RollingCompTransform(const T *data, int n, T *out, int window_size,
                                 int min_samples) {
  int upper_limit = std::min(window_size, n);
  SortedDeque<T, Comp> sdeque(window_size);
  for (int i = 0; i < upper_limit; ++i) {
    sdeque.update(data[i]);
    if (i + 1 < min_samples) {
      out[i] = std::numeric_limits<T>::quiet_NaN();
    } else {
      out[i] = sdeque.get();
    }
  }
  for (int i = upper_limit; i < n; ++i) {
    sdeque.update(data[i]);
    out[i] = sdeque.get();
  }
}

template <typename T>
void RollingMinTransform(const T *data, int n, T *out, int window_size,
                         int min_samples) {
  RollingCompTransform<T, std::greater_equal<T>>(data, n, out, window_size,
                                                 min_samples);
}

template <typename T>
void RollingMaxTransform(const T *data, int n, T *out, int window_size,
                         int min_samples) {
  RollingCompTransform<T, std::less_equal<T>>(data, n, out, window_size,
                                              min_samples);
}

template <typename T>
inline void RollingQuantileTransform(const T *data, int n, T *out,
                                     int window_size, int min_samples, T p) {
  int upper_limit = std::min(window_size, n);
  OrderedStructs::SkipList::HeadNode<T> sl;
  for (int i = 0; i < upper_limit; ++i) {
    sl.insert(data[i]);
    if (i + 1 < min_samples) {
      out[i] = std::numeric_limits<T>::quiet_NaN();
    } else {
      out[i] = SortedQuantile(sl, p, i + 1);
    }
  }
  for (int i = window_size; i < n; ++i) {
    sl.remove(data[i - window_size]);
    sl.insert(data[i]);
    out[i] = SortedQuantile(sl, p, window_size);
  }
}

template <typename Func, typename T, typename... Args>
inline void SeasonalRollingTransform(Func RollingTfm, const T *data, int n,
                                     T *out, int season_length, int window_size,
                                     int min_samples, Args &&...args) {
  int buff_size = n / season_length + (n % season_length > 0);
  T *season_data = new T[buff_size];
  T *season_out = new T[buff_size];
  std::fill_n(season_out, buff_size, std::numeric_limits<T>::quiet_NaN());
  for (int i = 0; i < season_length; ++i) {
    int season_n = n / season_length + (i < n % season_length);
    for (int j = 0; j < season_n; ++j) {
      season_data[j] = data[i + j * season_length];
    }
    RollingTfm(season_data, season_n, season_out, window_size, min_samples,
               std::forward<Args>(args)...);
    for (int j = 0; j < season_n; ++j) {
      out[i + j * season_length] = season_out[j];
    }
  }
  delete[] season_data;
  delete[] season_out;
}

template <typename T> struct SeasonalRollingMeanTransform {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingTransform(RollingMeanTransform<T>, data, n, out,
                             season_length, window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingStdTransform {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingTransform(RollingStdTransform<T>, data, n, out,
                             season_length, window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingMinTransform {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingTransform(RollingMinTransform<T>, data, n, out,
                             season_length, window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingMaxTransform {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingTransform(RollingMaxTransform<T>, data, n, out,
                             season_length, window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingQuantileTransform {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples, T p) {
    SeasonalRollingTransform(RollingQuantileTransform<T>, data, n, out,
                             season_length, window_size, min_samples, p);
  }
};

template <typename Func, typename T, typename... Args>
inline void RollingUpdate(Func RollingTfm, const T *data, int n, T *out,
                          int window_size, int min_samples, Args &&...args) {
  if (n < min_samples) {
    *out = std::numeric_limits<T>::quiet_NaN();
    return;
  }
  int n_samples = std::min(window_size, n);
  T *buffer = new T[n_samples];
  RollingTfm(data + n - n_samples, n_samples, buffer, window_size, min_samples,
             std::forward<Args>(args)...);
  *out = buffer[n_samples - 1];
  delete[] buffer;
}

template <typename T> struct RollingMeanUpdate {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples) {
    RollingUpdate(RollingMeanTransform<T>, data, n, out, window_size,
                  min_samples);
  }
};

template <typename T> struct RollingStdUpdate {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples) {
    RollingUpdate(RollingStdTransform<T>, data, n, out, window_size,
                  min_samples);
  }
};

template <typename T> struct RollingMinUpdate {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples) {
    RollingUpdate(RollingMinTransform<T>, data, n, out, window_size,
                  min_samples);
  }
};

template <typename T> struct RollingMaxUpdate {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples) {
    RollingUpdate(RollingMaxTransform<T>, data, n, out, window_size,
                  min_samples);
  }
};

template <typename T> struct RollingQuantileUpdate {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples, T p) {
    RollingUpdate(RollingQuantileTransform<T>, data, n, out, window_size,
                  min_samples, p);
  }
};

template <typename Func, typename T, typename... Args>
inline void SeasonalRollingUpdate(Func RollingUpdate, const T *data, int n,
                                  T *out, int season_length, int window_size,
                                  int min_samples, Args &&...args) {
  int season = n % season_length;
  int season_n = n / season_length + (season > 0);
  if (season_n < min_samples) {
    *out = std::numeric_limits<T>::quiet_NaN();
    return;
  }
  int n_samples = std::min(window_size, season_n);
  T *season_data = new T[n_samples];
  for (int i = 0; i < n_samples; ++i) {
    season_data[i] = data[n - 1 - (n_samples - 1 - i) * season_length];
  }
  RollingUpdate(season_data, n_samples, out, window_size, min_samples,
                std::forward<Args>(args)...);
  delete[] season_data;
}

template <typename T> struct SeasonalRollingMeanUpdate {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingUpdate(RollingMeanUpdate<T>(), data, n, out, season_length,
                          window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingStdUpdate {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingUpdate(RollingStdUpdate<T>(), data, n, out, season_length,
                          window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingMinUpdate {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingUpdate(RollingMinUpdate<T>(), data, n, out, season_length,
                          window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingMaxUpdate {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingUpdate(RollingMaxUpdate<T>(), data, n, out, season_length,
                          window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingQuantileUpdate {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples, T p) {
    SeasonalRollingUpdate(RollingQuantileUpdate<T>(), data, n, out,
                          season_length, window_size, min_samples, p);
  }
};
