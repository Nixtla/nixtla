#pragma once

#include "grouped_array.h"
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
    RollingMinTransform<T>(data, n, out, n, 1);
  }
};

template <typename T> struct ExpandingMaxTransform {
  void operator()(const T *data, int n, T *out) {
    RollingMaxTransform<T>(data, n, out, n, 1);
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
