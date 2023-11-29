#include <algorithm>
#include <cmath>
#include <numeric>

#include "coreforecast.h"

template <typename T>
inline T CommonScalerTransform(T data, T scale, T offset) {
  return (data - offset) / scale;
}

template <typename T>
inline T CommonScalerInverseTransform(T data, T scale, T offset) {
  return data * scale + offset;
}

template <typename T> inline indptr_t FirstNotNaN(const T *data, indptr_t n) {
  indptr_t i = 0;
  while (std::isnan(data[i]) && i < n) {
    ++i;
  }
  return i;
}

template <typename T>
inline indptr_t FirstNotNaN(const T *data, indptr_t n, T *out) {
  indptr_t i = 0;
  while (std::isnan(data[i]) && i < n) {
    out[i] = std::numeric_limits<T>::quiet_NaN();
    ++i;
  }
  return i;
}

template <typename T>
inline void TakeFromGroups(const T *data, int n, T *out, int k) {
  if (k > n) {
    *out = std::numeric_limits<T>::quiet_NaN();
  } else {
    *out = data[n - 1 - k];
  }
}

template <typename T>
inline void MinMaxScalerStats(const T *data, int n, T *stats) {
  T min = std::numeric_limits<T>::infinity();
  T max = -std::numeric_limits<T>::infinity();
  for (int i = 0; i < n; ++i) {
    if (data[i] < min)
      min = data[i];
    if (data[i] > max)
      max = data[i];
  }
  stats[0] = min;
  stats[1] = max - min;
}

template <typename T>
inline void StandardScalerStats(const T *data, int n, T *stats) {
  double sum = std::accumulate(data, data + n, 0.0);
  double mean = sum / n;
  double sum_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    sum_sq += (data[i] - mean) * (data[i] - mean);
  }
  stats[0] = static_cast<T>(mean);
  stats[1] = static_cast<T>(sqrt(sum_sq / n));
}

template <typename T> inline T Quantile(T *data, T p, int n) {
  T i_plus_g = p * (n - 1);
  int i = static_cast<int>(i_plus_g);
  T g = i_plus_g - i;
  std::nth_element(data, data + i, data + n);
  T out = data[i];
  if (g > 0.0) {
    std::nth_element(data, data + i + 1, data + n);
    out += g * (data[i + 1] - out);
  }
  return out;
}

template <typename T> inline T SortedQuantile(T *data, T p, int n) {
  T i_plus_g = p * (n - 1);
  int i = static_cast<int>(i_plus_g);
  T g = i_plus_g - i;
  T out = data[i];
  if (g > 0.0) {
    out += g * (data[i + 1] - out);
  }
  return out;
}

template <typename T>
inline void RobustScalerIqrStats(const T *data, int n, T *stats) {
  T *buffer = new T[n];
  std::copy(data, data + n, buffer);
  T median = Quantile(buffer, static_cast<T>(0.5), n);
  T q1 = Quantile(buffer, static_cast<T>(0.25), n);
  T q3 = Quantile(buffer, static_cast<T>(0.75), n);
  stats[0] = median;
  stats[1] = q3 - q1;
  delete[] buffer;
}

template <typename T>
inline void RobustScalerMadStats(const T *data, int n, T *stats) {
  T *buffer = new T[n];
  std::copy(data, data + n, buffer);
  const T median = Quantile(buffer, static_cast<T>(0.5), n);
  for (int i = 0; i < n; ++i) {
    buffer[i] = std::abs(buffer[i] - median);
  }
  T mad = Quantile(buffer, static_cast<T>(0.5), n);
  stats[0] = median;
  stats[1] = mad;
  delete[] buffer;
}

template <typename T> inline void LagTransform(const T *data, int n, T *out) {
  std::copy(data, data + n, out);
}

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
  T *buffer = new T[upper_limit];
  int *positions = new int[upper_limit];
  min_samples = std::min(min_samples, upper_limit);
  for (int i = 0; i < min_samples - 1; ++i) {
    buffer[i] = data[i];
    positions[i] = i;
    out[i] = std::numeric_limits<T>::quiet_NaN();
  }
  if (min_samples > 2) {
    std::sort(buffer, buffer + min_samples - 2);
  }
  for (int i = min_samples - 1; i < upper_limit; ++i) {
    int idx = std::lower_bound(buffer, buffer + i, data[i]) - buffer;
    for (int j = 0; j < i - idx; ++j) {
      buffer[i - j] = buffer[i - j - 1];
      positions[i - j] = positions[i - j - 1];
    }
    buffer[idx] = data[i];
    positions[idx] = i;
    out[i] = SortedQuantile(buffer, p, i + 1);
  }
  for (int i = window_size; i < n; ++i) {
    int remove_idx =
        std::min_element(positions, positions + window_size) - positions;
    int idx;
    if (data[i] <= buffer[remove_idx]) {
      idx = std::lower_bound(buffer, buffer + remove_idx, data[i]) - buffer;
      for (int j = 0; j < remove_idx - idx; ++j) {
        buffer[remove_idx - j] = buffer[remove_idx - j - 1];
        positions[remove_idx - j] = positions[remove_idx - j - 1];
      }
    } else {
      idx = (std::lower_bound(buffer + remove_idx - 1, buffer + window_size,
                              data[i]) -
             buffer) -
            1;
      if (idx == window_size) {
        --idx;
      }
      for (int j = 0; j < idx - remove_idx; ++j) {
        buffer[remove_idx + j] = buffer[remove_idx + j + 1];
        positions[remove_idx + j] = positions[remove_idx + j + 1];
      }
    }
    buffer[idx] = data[i];
    positions[idx] = i;
    out[i] = SortedQuantile(buffer, p, window_size);
  }
  delete[] buffer;
  delete[] positions;
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

template <typename T>
inline void ExpandingMeanTransform(const T *data, int n, T *out, T *agg) {
  T accum = static_cast<T>(0.0);
  for (int i = 0; i < n; ++i) {
    accum += data[i];
    out[i] = accum / (i + 1);
  }
  *agg = static_cast<T>(n);
}

template <typename T>
inline void ExpandingStdTransform(const T *data, int n, T *out, T *agg) {
  RollingStdTransformWithStats(data, n, out, agg, true, n, 2);
}

template <typename T> struct ExpandingMinTransform {
  void operator()(const T *data, int n, T *out) {
    RollingCompTransform(std::less<T>(), data, n, out, n, 1);
  }
};

template <typename T> struct ExpandingMaxTransform {
  void operator()(const T *data, int n, T *out) {
    RollingCompTransform(std::greater<T>(), data, n, out, n, 1);
  }
};

template <typename T>
inline void ExpandingQuantileTransform(const T *data, int n, T *out, T p) {
  RollingQuantileTransform(data, n, out, n, 1, p);
}

template <typename T>
inline void ExpandingQuantileUpdate(const T *data, int n, T *out, T p) {
  T *buffer = new T[n];
  std::copy(data, data + n, buffer);
  *out = Quantile(buffer, p, n);
  delete[] buffer;
}

template <typename T>
inline void ExponentiallyWeightedMeanTransform(const T *data, int n, T *out,
                                               T alpha) {
  out[0] = data[0];
  for (int i = 1; i < n; ++i) {
    out[i] = alpha * data[i] + (1 - alpha) * out[i - 1];
  }
}

template <typename T> inline void SkipLags(T *out, int n, int lag) {
  int replacements = std::min(lag, n);
  for (int i = 0; i < replacements; ++i) {
    out[i] = std::numeric_limits<T>::quiet_NaN();
  }
}

template <class T> class GroupedArray {
private:
  const T *data_;
  indptr_t n_data_;
  const indptr_t *indptr_;
  int n_groups_;
  int num_threads_;

public:
  GroupedArray(const T *data, indptr_t n_data, const indptr_t *indptr,
               int n_indptr, int num_threads)
      : data_(data), n_data_(n_data), indptr_(indptr), n_groups_(n_indptr - 1),
        num_threads_(num_threads) {}
  ~GroupedArray() {}
  template <typename Func, typename... Args>
  void Reduce(Func f, int n_out, T *out, int lag,
              Args &&...args) const noexcept {
#pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (int i = 0; i < n_groups_; ++i) {
      indptr_t start = indptr_[i];
      indptr_t end = indptr_[i + 1];
      indptr_t n = end - start;
      indptr_t start_idx = FirstNotNaN(data_ + start, n);
      if (start_idx + lag >= n)
        continue;
      start += start_idx;
      f(data_ + start, n - start_idx - lag, out + n_out * i,
        std::forward<Args>(args)...);
    }
  }

  template <typename Func>
  void ScalerTransform(Func f, const T *stats, T *out) const noexcept {
#pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (int i = 0; i < n_groups_; ++i) {
      indptr_t start = indptr_[i];
      indptr_t end = indptr_[i + 1];
      T offset = stats[2 * i];
      T scale = stats[2 * i + 1];
      if (std::abs(scale) < std::numeric_limits<T>::epsilon()) {
        scale = static_cast<T>(1.0);
      }
      for (indptr_t j = start; j < end; ++j) {
        out[j] = f(data_[j], scale, offset);
      }
    }
  }

  template <typename Func, typename... Args>
  void Transform(Func f, int lag, T *out, Args &&...args) const noexcept {
#pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (int i = 0; i < n_groups_; ++i) {
      indptr_t start = indptr_[i];
      indptr_t end = indptr_[i + 1];
      indptr_t n = end - start;
      indptr_t start_idx = FirstNotNaN(data_ + start, n, out + start);
      SkipLags(out + start + start_idx, n - start_idx, lag);
      if (start_idx + lag >= n) {
        continue;
      }
      start += start_idx;
      f(data_ + start, n - start_idx - lag, out + start + lag,
        std::forward<Args>(args)...);
    }
  }

  template <typename Func, typename... Args>
  void TransformAndReduce(Func f, int lag, T *out, int n_agg, T *agg,
                          Args &&...args) const noexcept {
#pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (int i = 0; i < n_groups_; ++i) {
      indptr_t start = indptr_[i];
      indptr_t end = indptr_[i + 1];
      indptr_t n = end - start;
      indptr_t start_idx = FirstNotNaN(data_ + start, n, out + start);
      SkipLags(out + start + start_idx, n - start_idx, lag);
      if (start_idx + lag >= n) {
        continue;
      }
      start += start_idx;
      f(data_ + start, n - start_idx - lag, out + start + lag, agg + i * n_agg,
        std::forward<Args>(args)...);
    }
  }
};

int GroupedArrayFloat32_Create(const float *data, indptr_t n_data,
                               indptr_t *indptr, indptr_t n_indptr,
                               int num_threads, GroupedArrayHandle *out) {
  *out = new GroupedArray<float>(data, n_data, indptr, n_indptr, num_threads);
  return 0;
}
int GroupedArrayFloat64_Create(const double *data, indptr_t n_data,
                               indptr_t *indptr, indptr_t n_indptr,
                               int num_threads, GroupedArrayHandle *out) {
  *out = new GroupedArray<double>(data, n_data, indptr, n_indptr, num_threads);
  return 0;
}

int GroupedArrayFloat32_Delete(GroupedArrayHandle handle) {
  delete reinterpret_cast<GroupedArray<float> *>(handle);
  return 0;
}
int GroupedArrayFloat64_Delete(GroupedArrayHandle handle) {
  delete reinterpret_cast<GroupedArray<double> *>(handle);
  return 0;
}

int GroupedArrayFloat32_MinMaxScalerStats(GroupedArrayHandle handle,
                                          float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(MinMaxScalerStats<float>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat64_MinMaxScalerStats(GroupedArrayHandle handle,
                                          double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(MinMaxScalerStats<double>, 2, out, 0);
  return 0;
}

int GroupedArrayFloat32_StandardScalerStats(GroupedArrayHandle handle,
                                            float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(StandardScalerStats<float>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat64_StandardScalerStats(GroupedArrayHandle handle,
                                            double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(StandardScalerStats<double>, 2, out, 0);
  return 0;
}

int GroupedArrayFloat32_RobustIqrScalerStats(GroupedArrayHandle handle,
                                             float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RobustScalerIqrStats<float>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat64_RobustIqrScalerStats(GroupedArrayHandle handle,
                                             double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RobustScalerIqrStats<double>, 2, out, 0);
  return 0;
}

int GroupedArrayFloat32_RobustMadScalerStats(GroupedArrayHandle handle,
                                             float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RobustScalerMadStats<float>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat64_RobustMadScalerStats(GroupedArrayHandle handle,
                                             double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RobustScalerMadStats<double>, 2, out, 0);
  return 0;
}

int GroupedArrayFloat32_ScalerTransform(GroupedArrayHandle handle,
                                        const float *stats, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->ScalerTransform(CommonScalerTransform<float>, stats, out);
  return 0;
}
int GroupedArrayFloat64_ScalerTransform(GroupedArrayHandle handle,
                                        const double *stats, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->ScalerTransform(CommonScalerTransform<double>, stats, out);
  return 0;
}

int GroupedArrayFloat32_ScalerInverseTransform(GroupedArrayHandle handle,
                                               const float *stats, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->ScalerTransform(CommonScalerInverseTransform<float>, stats, out);
  return 0;
}
int GroupedArrayFloat64_ScalerInverseTransform(GroupedArrayHandle handle,
                                               const double *stats,
                                               double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->ScalerTransform(CommonScalerInverseTransform<double>, stats, out);
  return 0;
}

int GroupedArrayFloat32_TakeFromGroups(GroupedArrayHandle handle, int k,
                                       float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(TakeFromGroups<float>, 1, out, 0, k);
  return 0;
}
int GroupedArrayFloat64_TakeFromGroups(GroupedArrayHandle handle, int k,
                                       double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(TakeFromGroups<double>, 1, out, 0, k);
  return 0;
}

int GroupedArrayFloat32_LagTransform(GroupedArrayHandle handle, int lag,
                                     float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(LagTransform<float>, lag, out);
  return 0;
}
int GroupedArrayFloat64_LagTransform(GroupedArrayHandle handle, int lag,
                                     double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(LagTransform<double>, lag, out);
  return 0;
}

int GroupedArrayFloat32_RollingMeanTransform(GroupedArrayHandle handle, int lag,
                                             int window_size, int min_samples,
                                             float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(RollingMeanTransform<float>, lag, out, window_size,
                min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingMeanTransform(GroupedArrayHandle handle, int lag,
                                             int window_size, int min_samples,
                                             double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(RollingMeanTransform<double>, lag, out, window_size,
                min_samples);
  return 0;
}

int GroupedArrayFloat32_RollingStdTransform(GroupedArrayHandle handle, int lag,
                                            int window_size, int min_samples,
                                            float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(RollingStdTransform<float>, lag, out, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingStdTransform(GroupedArrayHandle handle, int lag,
                                            int window_size, int min_samples,
                                            double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(RollingStdTransform<double>, lag, out, window_size,
                min_samples);
  return 0;
}

int GroupedArrayFloat32_RollingMinTransform(GroupedArrayHandle handle, int lag,
                                            int window_size, int min_samples,
                                            float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(RollingMinTransform<float>(), lag, out, window_size,
                min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingMinTransform(GroupedArrayHandle handle, int lag,
                                            int window_size, int min_samples,
                                            double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(RollingMinTransform<double>(), lag, out, window_size,
                min_samples);
  return 0;
}

int GroupedArrayFloat32_RollingMaxTransform(GroupedArrayHandle handle, int lag,
                                            int window_size, int min_samples,
                                            float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(RollingMaxTransform<float>(), lag, out, window_size,
                min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingMaxTransform(GroupedArrayHandle handle, int lag,
                                            int window_size, int min_samples,
                                            double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(RollingMaxTransform<double>(), lag, out, window_size,
                min_samples);
  return 0;
}

int GroupedArrayFloat32_RollingQuantileTransform(GroupedArrayHandle handle,
                                                 int lag, float p,
                                                 int window_size,
                                                 int min_samples, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(RollingQuantileTransform<float>, lag, out, window_size,
                min_samples, p);
  return 0;
}
int GroupedArrayFloat64_RollingQuantileTransform(GroupedArrayHandle handle,
                                                 int lag, double p,
                                                 int window_size,
                                                 int min_samples, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(RollingQuantileTransform<double>, lag, out, window_size,
                min_samples, p);
  return 0;
}

int GroupedArrayFloat32_RollingMeanUpdate(GroupedArrayHandle handle, int lag,
                                          int window_size, int min_samples,
                                          float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RollingMeanUpdate<float>(), 1, out, lag, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingMeanUpdate(GroupedArrayHandle handle, int lag,
                                          int window_size, int min_samples,
                                          double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RollingMeanUpdate<double>(), 1, out, lag, window_size,
             min_samples);
  return 0;
}

int GroupedArrayFloat32_RollingStdUpdate(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RollingStdUpdate<float>(), 1, out, lag, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingStdUpdate(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RollingStdUpdate<double>(), 1, out, lag, window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_RollingMinUpdate(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RollingMinUpdate<float>(), 1, out, lag, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingMinUpdate(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RollingMinUpdate<double>(), 1, out, lag, window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_RollingMaxUpdate(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RollingMaxUpdate<float>(), 1, out, lag, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingMaxUpdate(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RollingMaxUpdate<double>(), 1, out, lag, window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_RollingQuantileUpdate(GroupedArrayHandle handle,
                                              int lag, float p, int window_size,
                                              int min_samples, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RollingQuantileUpdate<float>(), 1, out, lag, window_size,
             min_samples, p);
  return 0;
}
int GroupedArrayFloat64_RollingQuantileUpdate(GroupedArrayHandle handle,
                                              int lag, double p,
                                              int window_size, int min_samples,
                                              double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RollingQuantileUpdate<double>(), 1, out, lag, window_size,
             min_samples, p);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingMeanTransform(GroupedArrayHandle handle,
                                                     int lag, int season_length,
                                                     int window_size,
                                                     int min_samples,
                                                     float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(SeasonalRollingMeanTransform<float>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingMeanTransform(GroupedArrayHandle handle,
                                                     int lag, int season_length,
                                                     int window_size,
                                                     int min_samples,
                                                     double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(SeasonalRollingMeanTransform<double>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingStdTransform(GroupedArrayHandle handle,
                                                    int lag, int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(SeasonalRollingStdTransform<float>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingStdTransform(GroupedArrayHandle handle,
                                                    int lag, int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(SeasonalRollingStdTransform<double>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingMinTransform(GroupedArrayHandle handle,
                                                    int lag, int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(SeasonalRollingMinTransform<float>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingMinTransform(GroupedArrayHandle handle,
                                                    int lag, int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(SeasonalRollingMinTransform<double>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingMaxTransform(GroupedArrayHandle handle,
                                                    int lag, int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(SeasonalRollingMaxTransform<float>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingMaxTransform(GroupedArrayHandle handle,
                                                    int lag, int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(SeasonalRollingMaxTransform<double>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingQuantileTransform(
    GroupedArrayHandle handle, int lag, int season_length, float p,
    int window_size, int min_samples, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(SeasonalRollingQuantileTransform<float>(), lag, out,
                season_length, window_size, min_samples, p);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingQuantileTransform(
    GroupedArrayHandle handle, int lag, int season_length, double p,
    int window_size, int min_samples, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(SeasonalRollingQuantileTransform<double>(), lag, out,
                season_length, window_size, min_samples, p);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingMeanUpdate(GroupedArrayHandle handle,
                                                  int lag, int season_length,
                                                  int window_size,
                                                  int min_samples, float *out) {

  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(SeasonalRollingMeanUpdate<float>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingMeanUpdate(GroupedArrayHandle handle,
                                                  int lag, int season_length,
                                                  int window_size,
                                                  int min_samples,
                                                  double *out) {

  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(SeasonalRollingMeanUpdate<double>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingStdUpdate(GroupedArrayHandle handle,
                                                 int lag, int season_length,
                                                 int window_size,
                                                 int min_samples, float *out) {

  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(SeasonalRollingStdUpdate<float>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingStdUpdate(GroupedArrayHandle handle,
                                                 int lag, int season_length,
                                                 int window_size,
                                                 int min_samples, double *out) {

  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(SeasonalRollingStdUpdate<double>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingMinUpdate(GroupedArrayHandle handle,
                                                 int lag, int season_length,
                                                 int window_size,
                                                 int min_samples, float *out) {

  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(SeasonalRollingMinUpdate<float>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingMinUpdate(GroupedArrayHandle handle,
                                                 int lag, int season_length,
                                                 int window_size,
                                                 int min_samples, double *out) {

  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(SeasonalRollingMinUpdate<double>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingMaxUpdate(GroupedArrayHandle handle,
                                                 int lag, int season_length,
                                                 int window_size,
                                                 int min_samples, float *out) {

  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(SeasonalRollingMaxUpdate<float>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingMaxUpdate(GroupedArrayHandle handle,
                                                 int lag, int season_length,
                                                 int window_size,
                                                 int min_samples, double *out) {

  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(SeasonalRollingMaxUpdate<double>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingQuantileUpdate(
    GroupedArrayHandle handle, int lag, int season_length, float p,
    int window_size, int min_samples, float *out) {

  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(SeasonalRollingQuantileUpdate<float>(), 1, out, lag, season_length,
             window_size, min_samples, p);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingQuantileUpdate(
    GroupedArrayHandle handle, int lag, int season_length, double p,
    int window_size, int min_samples, double *out) {

  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(SeasonalRollingQuantileUpdate<double>(), 1, out, lag,
             season_length, window_size, min_samples, p);
  return 0;
}

int GroupedArrayFloat32_ExpandingMeanTransform(GroupedArrayHandle handle,
                                               int lag, float *out,
                                               float *agg) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->TransformAndReduce(ExpandingMeanTransform<float>, lag, out, 1, agg);
  return 0;
}
int GroupedArrayFloat64_ExpandingMeanTransform(GroupedArrayHandle handle,
                                               int lag, double *out,
                                               double *agg) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->TransformAndReduce(ExpandingMeanTransform<double>, lag, out, 1, agg);
  return 0;
}

int GroupedArrayFloat32_ExpandingStdTransform(GroupedArrayHandle handle,
                                              int lag, float *out, float *agg) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->TransformAndReduce(ExpandingStdTransform<float>, lag, out, 3, agg);
  return 0;
}
int GroupedArrayFloat64_ExpandingStdTransform(GroupedArrayHandle handle,
                                              int lag, double *out,
                                              double *agg) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->TransformAndReduce(ExpandingStdTransform<double>, lag, out, 3, agg);
  return 0;
}

int GroupedArrayFloat32_ExpandingMinTransform(GroupedArrayHandle handle,
                                              int lag, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(ExpandingMinTransform<float>(), lag, out);
  return 0;
}
int GroupedArrayFloat64_ExpandingMinTransform(GroupedArrayHandle handle,
                                              int lag, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(ExpandingMinTransform<double>(), lag, out);
  return 0;
}

int GroupedArrayFloat32_ExpandingMaxTransform(GroupedArrayHandle handle,
                                              int lag, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(ExpandingMaxTransform<float>(), lag, out);

  return 0;
}
int GroupedArrayFloat64_ExpandingMaxTransform(GroupedArrayHandle handle,
                                              int lag, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(ExpandingMaxTransform<double>(), lag, out);

  return 0;
}

int GroupedArrayFloat32_ExpandingQuantileTransform(GroupedArrayHandle handle,
                                                   int lag, float p,
                                                   float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(ExpandingQuantileTransform<float>, lag, out, p);
  return 0;
}
int GroupedArrayFloat64_ExpandingQuantileTransform(GroupedArrayHandle handle,
                                                   int lag, double p,
                                                   double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(ExpandingQuantileTransform<double>, lag, out, p);
  return 0;
}

int GroupedArrayFloat32_ExpandingQuantileUpdate(GroupedArrayHandle handle,
                                                int lag, float p, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(ExpandingQuantileUpdate<float>, 1, out, lag, p);
  return 0;
}
int GroupedArrayFloat64_ExpandingQuantileUpdate(GroupedArrayHandle handle,
                                                int lag, double p,
                                                double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(ExpandingQuantileUpdate<double>, 1, out, lag, p);
  return 0;
}

int GroupedArrayFloat32_ExponentiallyWeightedMeanTransform(
    GroupedArrayHandle handle, int lag, float alpha, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(ExponentiallyWeightedMeanTransform<float>, lag, out, alpha);
  return 0;
}
int GroupedArrayFloat64_ExponentiallyWeightedMeanTransform(
    GroupedArrayHandle handle, int lag, double alpha, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(ExponentiallyWeightedMeanTransform<double>, lag, out, alpha);
  return 0;
}
