#pragma once

#include "stats.h"

#include <cmath>
#include <numeric>
#include <vector>

template <typename T> T KPSS(const T *x, size_t n, size_t lags) {
  T mean = Mean(x, n);
  auto resids = std::vector<T>(n);
  auto cresids = std::vector<T>(n);
  resids[0] = x[0] - mean;
  cresids[0] = resids[0];
  for (size_t i = 1; i < n; ++i) {
    resids[i] = x[i] - mean;
    cresids[i] = cresids[i - 1] + resids[i];
  }
  T eta = Dot(cresids.data(), cresids.data(), n) / (n * n);
  T s = std::accumulate(resids.begin(), resids.end(), 0.0,
                        [](T acc, T x) { return acc + x * x; });
  for (size_t i = 1; i < lags + 1; ++i) {
    T tmp = Dot(resids.data(), resids.data() + i, n - i);
    s += 2 * tmp * (1.0 - (i / (lags + 1.0)));
  }
  return n * eta / s;
}
