#pragma once

namespace exponentially_weighted {
template <typename T>
inline void MeanTransform(const T *data, int n, T *out, T alpha,
                          bool skipna = false) {
  if (!skipna) {
    // Fast path: original implementation
    out[0] = data[0];
    for (int i = 1; i < n; ++i) {
      out[i] = alpha * data[i] + (1 - alpha) * out[i - 1];
    }
  } else {
    // NaN-aware implementation with forward-fill behavior
    int first_valid = -1;
    for (int i = 0; i < n; ++i) {
      if (!std::isnan(data[i])) {
        first_valid = i;
        break;
      }
    }
    if (first_valid == -1) {
      // All values are NaN
      for (int i = 0; i < n; ++i) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
      }
      return;
    }
    // Fill leading NaNs
    for (int i = 0; i < first_valid; ++i) {
      out[i] = std::numeric_limits<T>::quiet_NaN();
    }
    // Initialize with first valid value
    out[first_valid] = data[first_valid];
    // Process remaining values
    for (int i = first_valid + 1; i < n; ++i) {
      if (!std::isnan(data[i])) {
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1];
      } else {
        // Forward-fill: use previous exponentially weighted mean
        out[i] = out[i - 1];
      }
    }
  }
}
} // namespace exponentially_weighted
