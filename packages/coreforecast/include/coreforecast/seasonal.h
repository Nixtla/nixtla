#pragma once

#include "common.h"
#include "stl.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

#include "stats.h"

namespace seasonal {
template <typename T>
void Difference(std::span<const T> data, std::span<T> out, index_t d) {
  const index_t n = std::ssize(data);
  if (d == 0) {
    std::copy(data.begin(), data.end(), out.begin());
    return;
  }
  if (n < d) {
    FillNaN(out);
    return;
  }
  FillNaN(out.first(d));
  for (index_t i = d; i < n; ++i) {
    out[i] = data[i] - data[i - d];
  }
}

template <typename T> T SeasHeuristic(std::span<const T> x, index_t period) {
  constexpr size_t seasonal = 11;
  size_t trend_length =
      static_cast<size_t>(std::ceil(1.5 * period / (1.0 - 1.5 / seasonal)));
  trend_length += trend_length % 2 == 0;
  size_t low_pass = period + (period % 2 == 0);
  stl::StlResult stl_fit = stl::params<T>()
                               .seasonal_length(seasonal)
                               .trend_length(trend_length)
                               .low_pass_length(low_pass)
                               .seasonal_degree(0)
                               .trend_degree(1)
                               .low_pass_degree(1)
                               .seasonal_jump(1)
                               .trend_jump(1)
                               .low_pass_jump(1)
                               .inner_loops(5)
                               .outer_loops(0)
                               .robust(false)
                               .fit(x.data(), x.size(), period);
  return stl_fit.seasonal_strength();
}

// The lag in [2, max_lag] with the largest autocovariance of the first
// differences, or 0 when there is nothing to work with. out has one element.
template <typename T>
void GreatestAutocovariance(std::span<const T> x, std::span<T> out,
                            index_t max_lag) {
  std::vector<T> diffs(x.size());
  Difference(x, std::span<T>{diffs}, 1);
  const std::span<const T> all{diffs};
  const std::span<const T> resids = all.subspan(FirstNotNaN(all));
  const index_t n = std::ssize(resids);
  if (n == 0) {
    out[0] = T{0};
    return;
  }
  max_lag = std::min(max_lag, n - 1);
  std::pair<T, index_t> result{-std::numeric_limits<T>::infinity(), 0};
  for (index_t i = 2; i < max_lag + 1; ++i) {
    const T cov = stats::Dot(resids.first(n - i), resids.subspan(i));
    if (cov > result.first) {
      result = {cov, i};
    }
  }
  out[0] = static_cast<T>(result.second);
}
} // namespace seasonal
