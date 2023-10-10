#include <algorithm>
#include <cmath>
#include <numeric>

#include "coreforecast.h"

inline float CommonScalerTransform(float data, double scale, double offset) {
  return static_cast<float>((data - offset) / scale);
}

inline float CommonScalerInverseTransform(float data, double scale,
                                          double offset) {
  return static_cast<float>(data * scale + offset);
}

inline int FirstNotNaN(const float *data, int n) {
  int i = 0;
  while (std::isnan(data[i]) && i < n) {
    ++i;
  }
  return i;
}

inline void MinMaxScalerStats(const float *data, int n, double *stats) {
  float min = std::numeric_limits<float>::infinity();
  float max = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < n; ++i) {
    if (data[i] < min)
      min = data[i];
    if (data[i] > max)
      max = data[i];
  }
  stats[0] = min;
  stats[1] = max - min;
}

inline void StandardScalerStats(const float *data, int n, double *stats) {
  double sum = std::accumulate(data, data + n, 0.0);
  double mean = sum / n;
  double sum_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    sum_sq += (data[i] - mean) * (data[i] - mean);
  }
  stats[0] = mean;
  stats[1] = sqrt(sum_sq / n);
}

inline double Quantile(float *data, float p, int n) {
  float i_plus_g = p * (n - 1);
  int i = int(i_plus_g);
  double g = i_plus_g - i;
  return data[i] + g * (data[i + 1] - data[i]);
}

inline void RobustScalerIqrStats(const float *data, int n, double *stats) {
  float *buffer = new float[n];
  std::copy(data, data + n, buffer);
  std::sort(buffer, buffer + n);
  double median = Quantile(buffer, 0.5F, n);
  double q1 = Quantile(buffer, 0.25F, n);
  double q3 = Quantile(buffer, 0.75F, n);
  stats[0] = median;
  stats[1] = q3 - q1;
  delete[] buffer;
}

inline void RobustScalerMadStats(const float *data, int n, double *stats) {
  float *buffer = new float[n];
  std::copy(data, data + n, buffer);
  std::sort(buffer, buffer + n);
  const float median = static_cast<float>(Quantile(buffer, 0.5F, n));
  for (int i = 0; i < n; ++i) {
    buffer[i] = std::abs(buffer[i] - median);
  }
  std::sort(buffer, buffer + n);
  double mad = Quantile(buffer, 0.5F, n);
  stats[0] = median;
  stats[1] = mad;
  delete[] buffer;
}

class GroupedArray {
private:
  float *data_;
  int32_t n_data_;
  int32_t *indptr_;
  int32_t n_groups_;

public:
  GroupedArray(float *data, int32_t n_data, int32_t *indptr, int32_t n_indptr)
      : data_(data), n_data_(n_data), indptr_(indptr), n_groups_(n_indptr - 1) {
  }
  ~GroupedArray() {}
  template <typename Func> void ComputeStats(Func f, double *out) const {
    for (int i = 0; i < n_groups_; ++i) {
      int32_t start = indptr_[i];
      int32_t end = indptr_[i + 1];
      int32_t n = end - start;
      int start_idx = FirstNotNaN(data_ + start, n);
      if (start_idx == n)
        continue;
      f(data_ + start + start_idx, n - start_idx, out + 2 * i);
    }
  }

  inline int32_t MaxGroupSize() const {
    int32_t max = 0;
    for (int i = 0; i < n_groups_; ++i) {
      int32_t grp_size = indptr_[i + 1] - indptr_[i];
      if (grp_size > max)
        max = grp_size;
    }
    return max;
  }

  template <typename Func>
  void ScalerTransform(Func f, const double *stats, float *out) const {
    for (int i = 0; i < n_groups_; ++i) {
      int32_t start = indptr_[i];
      int32_t end = indptr_[i + 1];
      double offset = stats[2 * i];
      double scale = stats[2 * i + 1];
      for (int32_t j = start; j < end; ++j) {
        out[j] = f(data_[j], scale, offset);
      }
    }
  }
};

int GroupedArray_CreateFromArrays(float *data, int32_t n_data, int32_t *indptr,
                                  int32_t n_groups, GroupedArrayHandle *out) {
  *out = new GroupedArray(data, n_data, indptr, n_groups);
  return 0;
}

int GroupedArray_Delete(GroupedArrayHandle handle) {
  delete reinterpret_cast<GroupedArray *>(handle);
  return 0;
}

int GroupedArray_MinMaxScalerStats(GroupedArrayHandle handle, double *out) {
  auto ga = reinterpret_cast<GroupedArray *>(handle);
  ga->ComputeStats(MinMaxScalerStats, out);
  return 0;
}

int GroupedArray_StandardScalerStats(GroupedArrayHandle handle, double *out) {
  auto ga = reinterpret_cast<GroupedArray *>(handle);
  ga->ComputeStats(StandardScalerStats, out);
  return 0;
}

int GroupedArray_RobustScalerIqrStats(GroupedArrayHandle handle, double *out) {
  auto ga = reinterpret_cast<GroupedArray *>(handle);
  ga->ComputeStats(RobustScalerIqrStats, out);
  return 0;
}

int GroupedArray_RobustScalerMadStats(GroupedArrayHandle handle, double *out) {
  auto ga = reinterpret_cast<GroupedArray *>(handle);
  ga->ComputeStats(RobustScalerMadStats, out);
  return 0;
}

int GroupedArray_ScalerTransform(GroupedArrayHandle handle, double *stats,
                                 float *out) {
  auto ga = reinterpret_cast<GroupedArray *>(handle);
  ga->ScalerTransform(CommonScalerTransform, stats, out);
  return 0;
}

int GroupedArray_ScalerInverseTransform(GroupedArrayHandle handle,
                                        double *stats, float *out) {
  auto ga = reinterpret_cast<GroupedArray *>(handle);
  ga->ScalerTransform(CommonScalerInverseTransform, stats, out);
  return 0;
}
