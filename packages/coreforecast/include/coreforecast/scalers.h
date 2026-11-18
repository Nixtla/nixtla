#pragma once

#include "brent.h"
#include "common.h"
#include "stats.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

namespace scalers {
template <typename T>
inline T CommonScalerTransform(T data, T offset, T scale) {
  return (data - offset) / scale;
}

template <typename T>
inline T CommonScalerInverseTransform(T data, T offset, T scale) {
  return data * scale + offset;
}

// Applies stats_fn to the data, or to a NaN-free copy of it when skipna is set,
// writing NaN stats when nothing valid remains. stats has two elements.
template <typename T, typename Fn>
inline void WithSkipNA(std::span<const T> data, std::span<T> stats, bool skipna,
                       Fn stats_fn) {
  if (!skipna) {
    stats_fn(data, stats);
    return;
  }
  std::vector<T> valid;
  valid.reserve(data.size());
  std::copy_if(data.begin(), data.end(), std::back_inserter(valid),
               [](T x) { return !std::isnan(x); });
  if (valid.empty()) {
    FillNaN(stats);
    return;
  }
  stats_fn(std::span<const T>{valid}, stats);
}

template <typename T>
inline void MinMaxScalerStats(std::span<const T> data, std::span<T> stats,
                              bool skipna = false) {
  WithSkipNA(data, stats, skipna, [](std::span<const T> d, std::span<T> s) {
    auto [min, max] = std::ranges::minmax(d);
    s[0] = min;
    s[1] = max - min;
  });
}

template <typename T>
inline void StandardScalerStats(std::span<const T> data, std::span<T> stats,
                                bool skipna = false) {
  WithSkipNA(data, stats, skipna, [](std::span<const T> d, std::span<T> s) {
    const double mean = stats::Mean(d);
    const double var = stats::SquaredDeviations(d, mean) / d.size();
    s[0] = static_cast<T>(mean);
    s[1] = static_cast<T>(std::sqrt(var));
  });
}

template <typename T>
inline void RobustScalerIqrStats(std::span<const T> data, std::span<T> stats,
                                 bool skipna = false) {
  WithSkipNA(data, stats, skipna, [](std::span<const T> d, std::span<T> s) {
    std::vector<T> buffer(d.begin(), d.end());
    const std::span<T> b{buffer};
    const T q1 = stats::Quantile(b, T{0.25});
    const T median = stats::Quantile(b, T{0.5});
    const T q3 = stats::Quantile(b, T{0.75});
    s[0] = median;
    s[1] = q3 - q1;
  });
}

template <typename T>
inline void RobustScalerMadStats(std::span<const T> data, std::span<T> stats,
                                 bool skipna = false) {
  WithSkipNA(data, stats, skipna, [](std::span<const T> d, std::span<T> s) {
    std::vector<T> buffer(d.begin(), d.end());
    const std::span<T> b{buffer};
    const T median = stats::Quantile(b, T{0.5});
    std::transform(buffer.begin(), buffer.end(), buffer.begin(),
                   [median](auto x) { return std::abs(x - median); });
    const T mad = stats::Quantile(b, T{0.5});
    s[0] = median;
    s[1] = mad;
  });
}

template <typename T>
T BoxCox_GuerreroCV(T lambda, const std::vector<T> &x_mean,
                    const std::vector<T> &x_std) {
  auto start_idx = size_t{0};
  for (const auto &x : x_std) {
    if (std::isnan(x)) {
      start_idx++;
    } else {
      break;
    }
  }
  if (x_std.size() - start_idx < 2) {
    return std::numeric_limits<T>::max();
  }
  const size_t m = x_std.size() - start_idx;
  std::vector<T> x_rat(m);
  for (size_t i = 0; i < m; ++i) {
    x_rat[i] = x_std[start_idx + i] /
               std::exp(std::log(x_mean[start_idx + i]) * (1 - lambda));
  }
  const double mean = stats::Mean<T>(x_rat);
  const double var = stats::SquaredDeviations<T>(x_rat, mean) / (m - 1);
  return static_cast<T>(std::sqrt(var) / mean);
}

// out has one element
template <typename T>
void BoxCoxLambdaGuerrero(std::span<const T> x, std::span<T> out,
                          index_t period, T lower, T upper) {
  assert(period > 0);
  const index_t n = std::ssize(x);
  if (n <= 2 * period) {
    out[0] = T{1.0};
    return;
  }
  if (std::any_of(x.begin(), x.end(), [](T v) { return v <= 0.0; })) {
    lower = std::max(lower, T{0.0});
  }
  const index_t n_seasons = n / period;
  const index_t n_full = n_seasons * period;
  // build matrix with subseries having full periods
  auto x_mat = std::vector<T>(n_seasons * period);
  std::copy(x.end() - n_full, x.end(), x_mat.begin());
  // means of subseries
  auto x_mean = std::vector<T>(n_seasons, 0.0);
  auto x_n = std::vector<index_t>(n_seasons, 0);
  for (index_t i = 0; i < n_seasons; ++i) {
    for (index_t j = 0; j < period; ++j) {
      if (std::isnan(x_mat[i * period + j])) {
        continue;
      }
      x_mean[i] += x_mat[i * period + j];
      x_n[i]++;
    }
    if (x_n[i] == 0) {
      x_mean[i] = kNaN<T>;
    } else {
      x_mean[i] /= x_n[i];
    }
  }
  // stds of subseries
  auto x_std = std::vector<T>(x_mean.size(), 0.0);
  for (index_t i = 0; i < std::ssize(x_std); ++i) {
    if (std::isnan(x_mean[i]) || x_n[i] < 2) {
      x_std[i] = kNaN<T>;
      continue;
    }
    for (index_t j = 0; j < period; ++j) {
      if (std::isnan(x_mat[i * period + j])) {
        continue;
      }
      T tmp = x_mat[i * period + j] - x_mean[i];
      x_std[i] += tmp * tmp;
    }
    x_std[i] = std::sqrt(x_std[i] / (x_n[i] - 1));
  }
  T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.25);
  out[0] = Brent(BoxCox_GuerreroCV<T>, lower, upper, tol, x_mean, x_std);
}

template <typename T> inline T BoxCoxTransform(T x, T lambda, T /*unused*/) {
  if (lambda < 0 && x < 0) {
    return kNaN<T>;
  }
  if (std::abs(lambda) < 1e-19) {
    return std::log(x);
  }
  if (x > 0) {
    return std::expm1(lambda * std::log(x)) / lambda;
  }
  return (-std::exp(lambda * std::log(-x)) - 1) / lambda;
}

template <typename T>
inline T BoxCoxInverseTransform(T x, T lambda, T /*unused*/) {
  if (lambda < 0 && lambda * x + 1 < 0) {
    return kNaN<T>;
  }
  if (lambda == 0) {
    return std::exp(x);
  }
  if (lambda * x + 1 > 0) {
    return std::exp(std::log1p(lambda * x) / lambda);
  }
  return -std::exp(std::log(-lambda * x - 1) / lambda);
}

template <typename T> T BoxCoxLogLik(T lambda, std::span<const T> data) {
  const index_t n = std::ssize(data);
  std::vector<double> logdata(n);
  for (index_t i = 0; i < n; ++i) {
    logdata[i] = std::log(static_cast<double>(data[i]));
  }
  const std::span<const double> logs{logdata};
  double var;
  if (lambda == 0.0) {
    var = stats::SquaredDeviations(logs, stats::Mean(logs)) / n;
  } else {
    std::vector<T> transformed(n);
    for (index_t i = 0; i < n; ++i) {
      transformed[i] = std::exp(std::log(data[i]) * lambda) / lambda;
    }
    const std::span<const T> tr{transformed};
    var = stats::SquaredDeviations(tr, stats::Mean(tr)) / n;
  }
  return -static_cast<T>((lambda - 1) * stats::Sum(logs) -
                         n / 2.0 * std::log(var));
}

// out has one element
template <typename T>
void BoxCoxLambdaLogLik(std::span<const T> x, std::span<T> out, T lower,
                        T upper) {
  T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.25);
  out[0] = Brent(BoxCoxLogLik<T>, lower, upper, tol, x);
}
} // namespace scalers
