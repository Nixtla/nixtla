#pragma once

#include "rolling.h"
#include "stats.h"

namespace expanding {
template <typename T>
inline void MeanTransform(const T *data, int n, T *out, T *agg,
                          bool skipna = false) {
  if (!skipna) {
    // Fast path: original implementation
    T accum = static_cast<T>(0.0);
    for (int i = 0; i < n; ++i) {
      accum += data[i];
      out[i] = accum / (i + 1);
    }
    *agg = static_cast<T>(n);
  } else {
    // NaN-aware implementation
    T accum = 0.0;
    int valid_count = 0;
    for (int i = 0; i < n; ++i) {
      if (!std::isnan(data[i])) {
        accum += data[i];
        valid_count++;
      }
      if (valid_count == 0) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
      } else {
        out[i] = accum / valid_count;
      }
    }
    *agg = static_cast<T>(valid_count);
  }
}

template <typename T>
inline void StdTransform(const T *data, int n, T *out, T *agg,
                         bool skipna = false) {
  rolling::StdTransformWithStats(data, n, out, agg, true, n, 2, skipna);
}

template <typename T>
inline void MinTransform(const T *data, int n, T *out, bool skipna = false) {
  rolling::MinTransform<T>(data, n, out, n, 1, skipna);
}

template <typename T>
inline void MaxTransform(const T *data, int n, T *out, bool skipna = false) {
  rolling::MaxTransform<T>(data, n, out, n, 1, skipna);
};

template <typename T>
inline void QuantileTransform(const T *data, int n, T *out, T p,
                              bool skipna = false) {
  rolling::QuantileTransform(data, n, out, n, 1, p, skipna);
}

template <typename T>
inline void QuantileUpdate(const T *data, int n, T *out, T p,
                           bool skipna = false) {
  if (!skipna) {
    std::vector<T> buffer(data, data + n);
    *out = stats::Quantile(buffer.begin(), buffer.end(), p);
  } else {
    std::vector<T> valid_data;
    for (int i = 0; i < n; ++i) {
      if (!std::isnan(data[i])) {
        valid_data.push_back(data[i]);
      }
    }
    if (valid_data.empty()) {
      *out = std::numeric_limits<T>::quiet_NaN();
    } else {
      *out = stats::Quantile(valid_data.begin(), valid_data.end(), p);
    }
  }
}
} // namespace expanding
