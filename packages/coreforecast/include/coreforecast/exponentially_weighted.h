#pragma once

namespace exponentially_weighted {
template <typename T>
inline void MeanTransform(const T *data, int n, T *out, T alpha) {
  out[0] = data[0];
  for (int i = 1; i < n; ++i) {
    out[i] = alpha * data[i] + (1 - alpha) * out[i - 1];
  }
}
} // namespace exponentially_weighted
