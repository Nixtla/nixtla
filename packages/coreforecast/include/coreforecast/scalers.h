#pragma once

#include "brent.h"
#include "stats.h"

#include <Eigen/Dense>

#include <algorithm>
#include <numeric>
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

template <typename T>
inline void MinMaxScalerStats(const T *data, int n, T *stats) {
  auto [min, max] = std::ranges::minmax(std::ranges::subrange(data, data + n));
  stats[0] = min;
  stats[1] = max - min;
}

template <typename T>
inline void StandardScalerStats(const T *data, int n, T *stats) {
  const Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> v(data, n);
  auto double_v = v.template cast<double>().array();
  double mean = double_v.mean();
  double std = std::sqrt((double_v - mean).square().mean());
  stats[0] = static_cast<T>(mean);
  stats[1] = static_cast<T>(std);
}

template <typename T>
inline void RobustScalerIqrStats(const T *data, int n, T *stats) {
  std::vector<T> buffer(data, data + n);
  const T q1 = stats::Quantile(buffer.begin(), buffer.end(), T{0.25});
  const T median = stats::Quantile(buffer.begin(), buffer.end(), T{0.5});
  const T q3 = stats::Quantile(buffer.begin(), buffer.end(), T{0.75});
  stats[0] = median;
  stats[1] = q3 - q1;
}

template <typename T>
inline void RobustScalerMadStats(const T *data, int n, T *stats) {
  std::vector<T> buffer(data, data + n);
  const T median = stats::Quantile(buffer.begin(), buffer.end(), T{0.5});
  std::transform(buffer.begin(), buffer.end(), buffer.begin(),
                 [median](auto x) { return std::abs(x - median); });
  const T mad = stats::Quantile(buffer.begin(), buffer.end(), T{0.5});
  stats[0] = median;
  stats[1] = mad;
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
  const Eigen::Map<const Eigen::VectorX<T>> mean_vec(x_mean.data() + start_idx,
                                                     x_mean.size() - start_idx);
  const Eigen::Map<const Eigen::VectorX<T>> std_vec(x_std.data() + start_idx,
                                                    x_std.size() - start_idx);
  auto x_rat =
      std_vec.array() / (mean_vec.array().log() * (1.0 - lambda)).exp();
  double mean = x_rat.mean();
  double std = (x_rat.array() - mean).square().sum() / (x_rat.size() - 1);
  return static_cast<T>(std / mean);
}

template <typename T>
void BoxCoxLambdaGuerrero(const T *x, int n, T *out, int period, T lower,
                          T upper) {
  if (n <= 2 * period) {
    *out = T{1.0};
    return;
  }
  for (int i = 0; i < n; ++i) {
    if (x[i] <= 0.0) {
      lower = std::max(lower, T{0.0});
      break;
    }
  }
  int n_seasons = n / period;
  int n_full = n_seasons * period;
  // build matrix with subseries having full periods
  auto x_mat = std::vector<T>(n_seasons * period);
  std::copy(x + n - n_full, x + n, x_mat.begin());
  // means of subseries
  auto x_mean = std::vector<T>(n_seasons, 0.0);
  auto x_n = std::vector<int>(n_seasons, 0);
  for (int i = 0; i < n_seasons; ++i) {
    for (int j = 0; j < period; ++j) {
      if (std::isnan(x_mat[i * period + j])) {
        continue;
      }
      x_mean[i] += x_mat[i * period + j];
      x_n[i]++;
    }
    x_mean[i] /= x_n[i];
  }
  // stds of subseries
  auto x_std = std::vector<T>(x_mean.size(), 0.0);
  for (size_t i = 0; i < x_std.size(); ++i) {
    if (std::isnan(x_mean[i]) || x_n[i] < 2) {
      x_std[i] = std::numeric_limits<T>::quiet_NaN();
      continue;
    }
    for (int j = 0; j < period; ++j) {
      if (std::isnan(x_mat[i * period + j])) {
        continue;
      }
      T tmp = x_mat[i * period + j] - x_mean[i];
      x_std[i] += tmp * tmp;
    }
    x_std[i] = std::sqrt(x_std[i] / (x_n[i] - 1));
  }
  T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.25);
  *out = Brent(BoxCox_GuerreroCV<T>, lower, upper, tol, x_mean, x_std);
}

template <typename T> inline T BoxCoxTransform(T x, T lambda, T /*unused*/) {
  if (lambda < 0 && x < 0) {
    return std::numeric_limits<T>::quiet_NaN();
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
    return std::numeric_limits<T>::quiet_NaN();
  }
  if (lambda == 0) {
    return std::exp(x);
  }
  if (lambda * x + 1 > 0) {
    return std::exp(std::log1p(lambda * x) / lambda);
  }
  return -std::exp(std::log(-lambda * x - 1) / lambda);
}

template <typename T> T BoxCoxLogLik(T lambda, const T *data, int n) {
  const Eigen::Map<const Eigen::VectorX<T>> v(data, n);
  const auto logdata = v.array().log().template cast<double>();
  double var;
  if (lambda == 0.0) {
    double mean = logdata.array().mean();
    var = (logdata.array() - mean).square().mean();
  } else {
    auto transformed = (v.array().log() * lambda).exp() / lambda;
    double mean = transformed.mean();
    var = (transformed - mean).square().mean();
  }
  return -static_cast<T>((lambda - 1) * logdata.sum() -
                         n / 2.0 * std::log(var));
}

template <typename T>
void BoxCoxLambdaLogLik(const T *x, int n, T *out, T lower, T upper) {
  T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.25);
  *out = Brent(BoxCoxLogLik<T>, lower, upper, tol, x, n);
}
} // namespace scalers
