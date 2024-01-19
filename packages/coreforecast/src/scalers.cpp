#include "scalers.h"
#include "stats.h"

#include <numeric>

template <typename T>
inline T CommonScalerTransform(T data, T scale, T offset) {
  return (data - offset) / scale;
}

template <typename T>
inline T CommonScalerInverseTransform(T data, T scale, T offset) {
  return data * scale + offset;
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
