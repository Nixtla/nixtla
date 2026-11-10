#pragma once

#include "SkipList.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <span>

#include "common.h"

namespace stats {
// Sum and dot product with four independent accumulators: the compiler can
// vectorise them without being allowed to reassociate, and the order (hence the
// result) is the same on every platform.
template <typename T> T Sum(std::span<const T> x) {
  const index_t n = std::ssize(x);
  T s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  index_t i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += x[i];
    s1 += x[i + 1];
    s2 += x[i + 2];
    s3 += x[i + 3];
  }
  for (; i < n; ++i) {
    s0 += x[i];
  }
  return (s0 + s1) + (s2 + s3);
}

template <typename T> T Dot(std::span<const T> a, std::span<const T> b) {
  assert(a.size() == b.size());
  const index_t n = std::ssize(a);
  T s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  index_t i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  for (; i < n; ++i) {
    s0 += a[i] * b[i];
  }
  return (s0 + s1) + (s2 + s3);
}

// Mean and population variance in double, whatever T is.
template <typename T> double Mean(std::span<const T> x) {
  double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  const index_t n = std::ssize(x);
  index_t i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += x[i];
    s1 += x[i + 1];
    s2 += x[i + 2];
    s3 += x[i + 3];
  }
  for (; i < n; ++i) {
    s0 += x[i];
  }
  return ((s0 + s1) + (s2 + s3)) / static_cast<double>(n);
}

template <typename T>
double SquaredDeviations(std::span<const T> x, double mean) {
  double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  const index_t n = std::ssize(x);
  index_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const double d0 = x[i] - mean, d1 = x[i + 1] - mean, d2 = x[i + 2] - mean,
                 d3 = x[i + 3] - mean;
    s0 += d0 * d0;
    s1 += d1 * d1;
    s2 += d2 * d2;
    s3 += d3 * d3;
  }
  for (; i < n; ++i) {
    const double d = x[i] - mean;
    s0 += d * d;
  }
  return (s0 + s1) + (s2 + s3);
}

// numpy's default linear interpolation. Partially sorts `data` in place.
template <typename T> T Quantile(std::span<T> data, T p) {
  const index_t n = std::ssize(data);
  const T i_plus_g = p * (n - 1);
  const auto i = static_cast<index_t>(i_plus_g);
  const T g = i_plus_g - i;

  auto nth = data.begin() + i;
  std::nth_element(data.begin(), nth, data.end());
  T out = *nth;

  if (g > T{0}) {
    auto min = std::min_element(nth + 1, data.end());
    out += g * (*min - out);
  }
  return out;
}

template <typename T>
T SortedQuantile(OrderedStructs::SkipList::HeadNode<T> &data, T p, index_t n) {
  const T i_plus_g = p * (n - 1);
  const auto i = static_cast<index_t>(i_plus_g);
  const T g = i_plus_g - i;
  T out = data.at(i);
  if (g > T{0.0}) {
    out += g * (data.at(i + 1) - out);
  }
  return out;
}
} // namespace stats
