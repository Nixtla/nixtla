#pragma once

#include "export.h"
#include "grouped_array.h"
#include "stats.h"

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

extern "C" {
DLL_EXPORT int
GroupedArrayFloat32_RollingMeanTransform(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         float *out);
DLL_EXPORT int
GroupedArrayFloat64_RollingMeanTransform(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         double *out);

DLL_EXPORT int
GroupedArrayFloat32_RollingStdTransform(GroupedArrayHandle handle, int lag,
                                        int window_size, int min_samples,
                                        float *out);
DLL_EXPORT int
GroupedArrayFloat64_RollingStdTransform(GroupedArrayHandle handle, int lag,
                                        int window_size, int min_samples,
                                        double *out);

DLL_EXPORT int
GroupedArrayFloat32_RollingMinTransform(GroupedArrayHandle handle, int lag,
                                        int window_size, int min_samples,
                                        float *out);
DLL_EXPORT int
GroupedArrayFloat64_RollingMinTransform(GroupedArrayHandle handle, int lag,
                                        int window_size, int min_samples,
                                        double *out);

DLL_EXPORT int
GroupedArrayFloat32_RollingMaxTransform(GroupedArrayHandle handle, int lag,
                                        int window_size, int min_samples,
                                        float *out);
DLL_EXPORT int
GroupedArrayFloat64_RollingMaxTransform(GroupedArrayHandle handle, int lag,
                                        int window_size, int min_samples,
                                        double *out);

DLL_EXPORT int
GroupedArrayFloat32_RollingQuantileTransform(GroupedArrayHandle handle, int lag,
                                             float p, int window_size,
                                             int min_samples, float *out);
DLL_EXPORT int
GroupedArrayFloat64_RollingQuantileTransform(GroupedArrayHandle handle, int lag,
                                             double p, int window_size,
                                             int min_samples, double *out);

DLL_EXPORT int GroupedArrayFloat32_RollingMeanUpdate(GroupedArrayHandle handle,
                                                     int lag, int window_size,
                                                     int min_samples,
                                                     float *out);
DLL_EXPORT int GroupedArrayFloat64_RollingMeanUpdate(GroupedArrayHandle handle,
                                                     int lag, int window_size,
                                                     int min_samples,
                                                     double *out);

DLL_EXPORT int GroupedArrayFloat32_RollingStdUpdate(GroupedArrayHandle handle,
                                                    int lag, int window_size,
                                                    int min_samples,
                                                    float *out);
DLL_EXPORT int GroupedArrayFloat64_RollingStdUpdate(GroupedArrayHandle handle,
                                                    int lag, int window_size,
                                                    int min_samples,
                                                    double *out);

DLL_EXPORT int GroupedArrayFloat32_RollingMinUpdate(GroupedArrayHandle handle,
                                                    int lag, int window_size,
                                                    int min_samples,
                                                    float *out);
DLL_EXPORT int GroupedArrayFloat64_RollingMinUpdate(GroupedArrayHandle handle,
                                                    int lag, int window_size,
                                                    int min_samples,
                                                    double *out);

DLL_EXPORT int GroupedArrayFloat32_RollingMaxUpdate(GroupedArrayHandle handle,
                                                    int lag, int window_size,
                                                    int min_samples,
                                                    float *out);
DLL_EXPORT int GroupedArrayFloat64_RollingMaxUpdate(GroupedArrayHandle handle,
                                                    int lag, int window_size,
                                                    int min_samples,
                                                    double *out);

DLL_EXPORT int
GroupedArrayFloat32_RollingQuantileUpdate(GroupedArrayHandle handle, int lag,
                                          float p, int window_size,
                                          int min_samples, float *out);
DLL_EXPORT int
GroupedArrayFloat64_RollingQuantileUpdate(GroupedArrayHandle handle, int lag,
                                          double p, int window_size,
                                          int min_samples, double *out);

DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingMeanTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingMeanTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, double *out);

DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingStdTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingStdTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, double *out);

DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingMinTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingMinTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, double *out);

DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingMaxTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingMaxTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, double *out);

DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingQuantileTransform(
    GroupedArrayHandle handle, int lag, int season_length, float p,
    int window_size, int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingQuantileTransform(
    GroupedArrayHandle handle, int lag, int season_length, double p,
    int window_size, int min_samples, double *out);

DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingMeanUpdate(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingMeanUpdate(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, double *out);

DLL_EXPORT int
GroupedArrayFloat32_SeasonalRollingStdUpdate(GroupedArrayHandle handle, int lag,
                                             int season_length, int window_size,
                                             int min_samples, float *out);
DLL_EXPORT int
GroupedArrayFloat64_SeasonalRollingStdUpdate(GroupedArrayHandle handle, int lag,
                                             int season_length, int window_size,
                                             int min_samples, double *out);

DLL_EXPORT int
GroupedArrayFloat32_SeasonalRollingMinUpdate(GroupedArrayHandle handle, int lag,
                                             int season_length, int window_size,
                                             int min_samples, float *out);
DLL_EXPORT int
GroupedArrayFloat64_SeasonalRollingMinUpdate(GroupedArrayHandle handle, int lag,
                                             int season_length, int window_size,
                                             int min_samples, double *out);

DLL_EXPORT int
GroupedArrayFloat32_SeasonalRollingMaxUpdate(GroupedArrayHandle handle, int lag,
                                             int season_length, int window_size,
                                             int min_samples, float *out);
DLL_EXPORT int
GroupedArrayFloat64_SeasonalRollingMaxUpdate(GroupedArrayHandle handle, int lag,
                                             int season_length, int window_size,
                                             int min_samples, double *out);

DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingQuantileUpdate(
    GroupedArrayHandle handle, int lag, int season_length, float p,
    int window_size, int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingQuantileUpdate(
    GroupedArrayHandle handle, int lag, int season_length, double p,
    int window_size, int min_samples, double *out);
}
