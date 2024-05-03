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

template <typename Func, typename T>
inline void RollingCompTransform(Func Comp, const T *data, int n, T *out,
                                 int window_size, int min_samples) {
  int upper_limit = std::min(window_size, n);
  T pivot = data[0];
  for (int i = 0; i < upper_limit; ++i) {
    if (Comp(data[i], pivot)) {
      pivot = data[i];
    }
    if (i + 1 < min_samples) {
      out[i] = std::numeric_limits<T>::quiet_NaN();
    } else {
      out[i] = pivot;
    }
  }
  for (int i = window_size; i < n; ++i) {
    pivot = data[i];
    for (int j = 0; j < window_size; ++j) {
      if (Comp(data[i - j], pivot)) {
        pivot = data[i - j];
      }
    }
    out[i] = pivot;
  }
}

template <typename T> struct RollingMinTransform {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples) {
    RollingCompTransform(std::less<T>(), data, n, out, window_size,
                         min_samples);
  }
};

template <typename T> struct RollingMaxTransform {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples) const {
    RollingCompTransform(std::greater<T>(), data, n, out, window_size,
                         min_samples);
  }
};

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
    SeasonalRollingTransform(RollingMinTransform<T>(), data, n, out,
                             season_length, window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingMaxTransform {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingTransform(RollingMaxTransform<T>(), data, n, out,
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
    RollingUpdate(RollingMinTransform<T>(), data, n, out, window_size,
                  min_samples);
  }
};

template <typename T> struct RollingMaxUpdate {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples) {
    RollingUpdate(RollingMaxTransform<T>(), data, n, out, window_size,
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
