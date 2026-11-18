#pragma once

#include <cmath>
#include <span>

#include "common.h"

namespace exponentially_weighted {
template <typename T>
inline void MeanTransform(std::span<const T> data, std::span<T> out, T alpha,
                          bool skipna = false) {
  const index_t n = std::ssize(data);
  if (!skipna) {
    out[0] = data[0];
    for (index_t i = 1; i < n; ++i) {
      out[i] = alpha * data[i] + (1 - alpha) * out[i - 1];
    }
    return;
  }
  // NaN-aware: a NaN forward-fills the previous mean
  const index_t first_valid = FirstNotNaN(data, out);
  if (first_valid == n) {
    return;
  }
  out[first_valid] = data[first_valid];
  for (index_t i = first_valid + 1; i < n; ++i) {
    if (std::isnan(data[i])) {
      out[i] = out[i - 1];
    } else {
      out[i] = alpha * data[i] + (1 - alpha) * out[i - 1];
    }
  }
}
} // namespace exponentially_weighted
