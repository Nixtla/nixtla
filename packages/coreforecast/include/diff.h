#pragma once

#include "grouped_array.h"
#include "kpss.h"
#include "seasonal.h"

template <typename T> void Differences(const T *data, int n, int d, T *out) {
  Difference(data, n, out, d);
}

template <typename T> inline bool IsConstant(const T *data, int n) {
  for (int i = 1; i < n; ++i) {
    if (data[i] != data[0]) {
      return false;
    }
  }
  return true;
}

template <typename T>
void InvertDifference(const T *data, int n, const T *tails, int d, T *out) {
  if (d == 0) {
    std::copy(data, data + n, out);
    return;
  }
  int upper = std::min(d, n);
  for (int i = 0; i < upper; ++i) {
    out[i] = data[i] + tails[i];
  }
  for (int i = upper; i < n; ++i) {
    out[i] = data[i] + out[i - d];
  }
}

template <typename T> void NumDiffs(const T *x, indptr_t n, T *out, int max_d) {
  // assume there are only NaNs at the start
  indptr_t start_idx = FirstNotNaN(x, n);
  x += start_idx;
  n -= start_idx;
  if (n < 3) {
    *out = 0;
    return;
  }
  constexpr T threshold = 0.463; // alpha = 0.05
  int d = 0;
  int n_lags = std::floor(3 * std::sqrt(n) / 13);
  bool do_diff = KPSS(x, n, n_lags) > threshold;
  std::vector<T> x_vec(n);
  std::copy(x, x + n, x_vec.begin());
  std::vector<T> diff_x(n);
  while (do_diff && d < max_d) {
    ++d;
    Difference(x_vec.data(), x_vec.size(), diff_x.data(), 1);
    if (IsConstant(diff_x.data() + d, diff_x.size() - d)) {
      *out = d;
      return;
    }
    std::copy(diff_x.begin(), diff_x.end(), x_vec.begin());
    if (n > d) {
      // we've taken d differences, so we have d NaNs
      do_diff = KPSS(x_vec.data() + d, n - d, n_lags) > threshold;
    } else {
      do_diff = false;
    }
  }
  *out = d;
  return;
}

template <typename T>
void NumSeasDiffs(const T *x, indptr_t n, T *out, int period, int max_d) {
  // assume there are only NaNs at the start
  indptr_t start_idx = FirstNotNaN(x, n);
  x += start_idx;
  n -= start_idx;
  if (n < 2 * period) {
    *out = 0;
    return;
  }
  constexpr T threshold = 0.64;
  int d = 0;
  bool do_diff = SeasHeuristic(x, n, period) > threshold;
  std::vector<T> x_vec(n);
  std::copy(x, x + n, x_vec.begin());
  std::vector<T> diff_x(n);
  while (do_diff && d < max_d) {
    ++d;
    Difference(x_vec.data(), x_vec.size(), diff_x.data(), period);
    if (IsConstant(diff_x.data() + d * period, n - d * period)) {
      *out = d;
      return;
    }
    std::copy(diff_x.begin(), diff_x.end(), x_vec.begin());
    // we'll have d * period NaNs and we need 2 * period samples for the STL
    if (n > (d + 2) * period && d < max_d) {
      do_diff = SeasHeuristic(x_vec.data() + d * period, n - d * period,
                              period) > threshold;
    } else {
      do_diff = false;
    }
  }
  *out = d;
  return;
}

template <typename T>
void NumSeasDiffsPeriods(const T *x, indptr_t n, T *period_and_out, int max_d) {
  int period = static_cast<int>(period_and_out[0]);
  T *out = period_and_out + 1;
  NumSeasDiffs(x, n, out, period, max_d);
}
