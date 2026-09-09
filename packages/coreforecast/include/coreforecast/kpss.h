#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <vector>

#include "common.h"
#include "stats.h"

// lags beyond n - 1 have no pairs to sum, so they are clamped like
// seasonal::GreatestAutocovariance does with max_lag.
template <typename T> T KPSS(std::span<const T> x, index_t lags) {
  const index_t n = std::ssize(x);
  lags = std::min(lags, n - 1);
  const T mean = stats::Sum(x) / static_cast<T>(n);
  std::vector<T> resids(n);
  for (index_t i = 0; i < n; ++i) {
    resids[i] = x[i] - mean;
  }
  std::vector<T> cresids(n);
  std::partial_sum(resids.begin(), resids.end(), cresids.begin());
  const std::span<const T> r{resids};
  // n * n would overflow index_t from 3e9 elements on
  const T eta =
      stats::Dot<T>(cresids, cresids) / (static_cast<T>(n) * static_cast<T>(n));
  T s = stats::Dot(r, r);
  for (index_t i = 1; i < lags + 1; ++i) {
    const T tmp = stats::Dot(r.first(n - i), r.subspan(i));
    s += 2 * tmp * static_cast<T>(1.0 - (i / (lags + 1.0)));
  }
  return static_cast<T>(n) * eta / s;
}
