#pragma once

template <typename T> inline void LagTransform(const T *data, int n, T *out) {
  std::copy(data, data + n, out);
}
