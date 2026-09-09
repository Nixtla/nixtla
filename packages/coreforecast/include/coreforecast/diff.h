#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <span>
#include <vector>

#include "common.h"
#include "kpss.h"
#include "seasonal.h"

namespace diff {
template <typename T>
void Differences(std::span<const T> data, std::span<T> out, index_t d) {
  seasonal::Difference(data, out, d);
}

// Vacuously true when empty, which NumDiffs reaches once d catches up with n.
template <typename T> inline bool IsConstant(std::span<const T> data) {
  if (data.empty()) {
    return true;
  }
  const T first = data[0];
  return std::all_of(data.begin(), data.end(),
                     [first](T x) { return x == first; });
}

// Undoes a difference of order tails.size(): the first d outputs need the d
// values that preceded the series, the rest build on the outputs before them.
template <typename T>
void InvertDifference(std::span<const T> data, std::span<const T> tails,
                      std::span<T> out) {
  const index_t n = std::ssize(data);
  const index_t d = std::ssize(tails);
  if (d == 0) {
    std::copy(data.begin(), data.end(), out.begin());
    return;
  }
  const index_t upper = std::min(d, n);
  for (index_t i = 0; i < upper; ++i) {
    out[i] = data[i] + tails[i];
  }
  for (index_t i = upper; i < n; ++i) {
    out[i] = data[i] + out[i - d];
  }
}

// out has one element
template <typename T>
void NumDiffs(std::span<const T> x, std::span<T> out, index_t max_d) {
  // assume there are only NaNs at the start
  x = x.subspan(FirstNotNaN(x));
  const index_t n = std::ssize(x);
  if (n < 3) {
    out[0] = 0;
    return;
  }
  constexpr T threshold = 0.463; // alpha = 0.05
  index_t d = 0;
  const index_t n_lags = std::floor(3 * std::sqrt(n) / 13);
  std::vector<T> x_vec(x.begin(), x.end());
  bool do_diff = KPSS(std::span<const T>{x_vec}, n_lags) > threshold;
  std::vector<T> diff_x(n);
  while (do_diff && d < max_d) {
    ++d;
    seasonal::Difference(std::span<const T>{x_vec}, std::span<T>{diff_x}, 1);
    if (IsConstant(std::span<const T>{diff_x}.subspan(d))) {
      out[0] = d;
      return;
    }
    std::copy(diff_x.begin(), diff_x.end(), x_vec.begin());
    if (n > d) {
      // we've taken d differences, so we have d NaNs
      do_diff = KPSS(std::span<const T>{x_vec}.subspan(d), n_lags) > threshold;
    } else {
      do_diff = false;
    }
  }
  out[0] = d;
}

// out has one element
template <typename T>
void NumSeasDiffs(std::span<const T> x, std::span<T> out, index_t period,
                  index_t max_d) {
  assert(period >= 0);
  // find_season_length passes the zero it gets when it finds no seasonality
  if (period == 0) {
    out[0] = 0;
    return;
  }
  // assume there are only NaNs at the start
  x = x.subspan(FirstNotNaN(x));
  const index_t n = std::ssize(x);
  if (n < 2 * period) {
    out[0] = 0;
    return;
  }
  constexpr T threshold = 0.64;
  index_t d = 0;
  bool do_diff = seasonal::SeasHeuristic(x, period) > threshold;
  std::vector<T> x_vec(x.begin(), x.end());
  std::vector<T> diff_x(n);
  while (do_diff && d < max_d) {
    ++d;
    seasonal::Difference(std::span<const T>{x_vec}, std::span<T>{diff_x},
                         period);
    if (IsConstant(std::span<const T>{diff_x}.subspan(d * period))) {
      out[0] = d;
      return;
    }
    std::copy(diff_x.begin(), diff_x.end(), x_vec.begin());
    // we'll have d * period NaNs and we need 2 * period samples for the STL
    if (n > (d + 2) * period && d < max_d) {
      do_diff =
          seasonal::SeasHeuristic(std::span<const T>{x_vec}.subspan(d * period),
                                  period) > threshold;
    } else {
      do_diff = false;
    }
  }
  out[0] = d;
}

// period_and_out holds the period on entry and the result on exit, so the
// per-group period can travel through Reduce's output buffer. The period is a
// float from Periods: NaN (no period for this group) gives a NaN count, and
// anything past the group length gives zero before the cast, which is
// otherwise undefined for NaN, infinity and values beyond index_t.
template <typename T>
void NumSeasDiffsPeriods(std::span<const T> x, std::span<T> period_and_out,
                         index_t max_d) {
  const T period = period_and_out[0];
  const auto out = period_and_out.subspan(1, 1);
  if (std::isnan(period)) {
    out[0] = kNaN<T>;
    return;
  }
  if (period > static_cast<T>(std::ssize(x))) {
    out[0] = 0; // NumSeasDiffs needs two periods
    return;
  }
  NumSeasDiffs(x, out, static_cast<index_t>(period), max_d);
}
} // namespace diff
