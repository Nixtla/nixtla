#include <algorithm>
#include <numeric>

#include "expanding.h"
#include "rolling.h"
#include "stats.h"

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
