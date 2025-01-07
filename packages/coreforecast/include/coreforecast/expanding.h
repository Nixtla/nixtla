#pragma once

#include "rolling.h"
#include "stats.h"

namespace expanding {
template <typename T>
inline void MeanTransform(const T *data, int n, T *out, T *agg) {
  T accum = static_cast<T>(0.0);
  for (int i = 0; i < n; ++i) {
    accum += data[i];
    out[i] = accum / (i + 1);
  }
  *agg = static_cast<T>(n);
}

template <typename T>
inline void StdTransform(const T *data, int n, T *out, T *agg) {
  rolling::StdTransformWithStats(data, n, out, agg, true, n, 2);
}

template <typename T> inline void MinTransform(const T *data, int n, T *out) {
  rolling::MinTransform<T>(data, n, out, n, 1);
}

template <typename T> inline void MaxTransform(const T *data, int n, T *out) {
  rolling::MaxTransform<T>(data, n, out, n, 1);
};

template <typename T>
inline void QuantileTransform(const T *data, int n, T *out, T p) {
  rolling::QuantileTransform(data, n, out, n, 1, p);
}

template <typename T>
inline void QuantileUpdate(const T *data, int n, T *out, T p) {
  std::vector<T> buffer(data, data + n);
  *out = stats::Quantile(buffer.begin(), buffer.end(), p);
}
} // namespace expanding
