#pragma once

#include "stats.h"
#include "stl.hpp"

#include <cmath>

template <typename T> void Difference(const T *data, int n, T *out, int d) {
  if (d == 0) {
    std::copy(data, data + n, out);
    return;
  }
  if (n < d) {
    std::fill(out, out + n, std::numeric_limits<T>::quiet_NaN());
    return;
  }
  std::fill(out, out + d, std::numeric_limits<T>::quiet_NaN());
  for (int i = d; i < n; ++i) {
    out[i] = data[i] - data[i - d];
  }
}

template <typename T> T SeasHeuristic(const T *x, size_t n, size_t period) {
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
                               .fit(x, n, period);
  return stl_fit.seasonal_strength();
}

template <typename T>
void GreatestAutocovariance(const T *x, size_t n, T *out, size_t max_lag) {
  T *resids = new T[n];
  Difference(x, n, resids, 1);
  indptr_t start = FirstNotNaN(resids, n);
  max_lag = std::min(max_lag, n - start - 1);
  T max_cov = -std::numeric_limits<T>::infinity();
  size_t lag = 0;
  for (size_t i = 2; i < max_lag + 1; ++i) {
    T cov = Dot(resids + start, resids + start + i, n - start - i);
    if (cov > max_cov) {
      max_cov = cov;
      lag = i;
    }
  }
  delete[] resids;
  *out = static_cast<T>(lag);
}
