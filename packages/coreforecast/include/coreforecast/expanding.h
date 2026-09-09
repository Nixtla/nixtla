#pragma once

#include <cassert>
#include <span>
#include <vector>

#include "rolling.h"
#include "stats.h"

// The expanding statistics are the rolling ones with the whole group as the
// window; that Window is built here from a group the driver already checked,
// so it does not go through Window::Checked.
namespace expanding {
// agg is empty or has one element: the number of values the last mean was
// taken over
template <typename T>
inline void MeanTransform(std::span<const T> data, std::span<T> out,
                          std::span<T> agg, bool skipna = false) {
  const index_t n = std::ssize(data);
  if (!skipna) {
    T accum = static_cast<T>(0.0);
    for (index_t i = 0; i < n; ++i) {
      accum += data[i];
      out[i] = accum / (i + 1);
    }
    if (!agg.empty()) {
      agg[0] = static_cast<T>(n);
    }
    return;
  }
  T accum = 0.0;
  index_t valid_count = 0;
  for (index_t i = 0; i < n; ++i) {
    if (!std::isnan(data[i])) {
      accum += data[i];
      valid_count++;
    }
    out[i] = valid_count == 0 ? kNaN<T> : accum / valid_count;
  }
  if (!agg.empty()) {
    agg[0] = static_cast<T>(valid_count);
  }
}

// agg has three elements, see rolling::StdTransformWithStats
template <typename T>
inline void StdTransform(std::span<const T> data, std::span<T> out,
                         std::span<T> agg, bool skipna = false) {
  rolling::StdTransformWithStats(data, out, agg,
                                 Window{std::ssize(data), 2, skipna});
}

template <typename T>
inline void MinTransform(std::span<const T> data, std::span<T> out,
                         bool skipna = false) {
  rolling::MinTransform<T>(data, out, Window{std::ssize(data), 1, skipna});
}

template <typename T>
inline void MaxTransform(std::span<const T> data, std::span<T> out,
                         bool skipna = false) {
  rolling::MaxTransform<T>(data, out, Window{std::ssize(data), 1, skipna});
}

template <typename T>
inline void QuantileTransform(std::span<const T> data, std::span<T> out, T p,
                              bool skipna = false) {
  rolling::QuantileTransform(data, out, Window{std::ssize(data), 1, skipna}, p);
}

template <typename T>
inline void QuantileUpdate(std::span<const T> data, std::span<T> out, T p,
                           bool skipna = false) {
  assert(p >= 0 && p <= 1);
  std::vector<T> buffer;
  if (!skipna) {
    buffer.assign(data.begin(), data.end());
  } else {
    std::copy_if(data.begin(), data.end(), std::back_inserter(buffer),
                 [](T x) { return !std::isnan(x); });
    if (buffer.empty()) {
      out[0] = kNaN<T>;
      return;
    }
  }
  out[0] = stats::Quantile(std::span<T>{buffer}, p);
}
} // namespace expanding
