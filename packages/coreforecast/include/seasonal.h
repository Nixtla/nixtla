#pragma once

#include "export.h"
#include "stl.hpp"

#include <cmath>

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

extern "C" {
DLL_EXPORT float Float32_SeasHeuristic(const float *x, size_t n,
                                       size_t period) {
  return SeasHeuristic(x, n, period);
}
DLL_EXPORT float Float64_SeasHeuristic(const double *x, size_t n,
                                       size_t period) {
  return SeasHeuristic(x, n, period);
}
}
