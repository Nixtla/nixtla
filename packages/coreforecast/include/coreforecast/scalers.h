#pragma once

#include "brent.h"
#include "stats.h"

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
  T min = std::numeric_limits<T>::infinity();
  T max = -std::numeric_limits<T>::infinity();
  for (int i = 0; i < n; ++i) {
    if (data[i] < min)
      min = data[i];
    if (data[i] > max)
      max = data[i];
  }
  stats[0] = min;
  stats[1] = max - min;
}

template <typename T>
inline void StandardScalerStats(const T *data, int n, T *stats) {
  double mean = Mean(data, n);
  double std = StandardDeviation(data, n, mean);
  stats[0] = static_cast<T>(mean);
  stats[1] = static_cast<T>(std);
}

template <typename T>
inline void RobustScalerIqrStats(const T *data, int n, T *stats) {
  std::vector<T> buffer(data, data + n);
  T median = Quantile(buffer.data(), static_cast<T>(0.5), n);
  T q1 = Quantile(buffer.data(), static_cast<T>(0.25), n);
  T q3 = Quantile(buffer.data(), static_cast<T>(0.75), n);
  stats[0] = median;
  stats[1] = q3 - q1;
}

template <typename T>
inline void RobustScalerMadStats(const T *data, int n, T *stats) {
  std::vector<T> buffer(data, data + n);
  const T median = Quantile(buffer.data(), static_cast<T>(0.5), n);
  for (int i = 0; i < n; ++i) {
    buffer[i] = std::abs(buffer[i] - median);
  }
  T mad = Quantile(buffer.data(), static_cast<T>(0.5), n);
  stats[0] = median;
  stats[1] = mad;
}

template <typename T>
T BoxCox_GuerreroCV(T lambda, const std::vector<T> &x_mean,
                    const std::vector<T> &x_std) {
  auto x_rat = std::vector<T>(x_std.size());
  int start_idx = 0;
  for (size_t i = 0; i < x_rat.size(); ++i) {
    if (std::isnan(x_std[i])) {
      start_idx++;
      continue;
    }
    x_rat[i] = x_std[i] / std::exp((1.0 - lambda) * std::log(x_mean[i]));
  }
  double mean = Mean(x_rat.data() + start_idx, x_rat.size() - start_idx);
  double std = StandardDeviation(x_rat.data() + start_idx,
                                 x_rat.size() - start_idx, mean, 1);
  if (std::isnan(std)) {
    return std::numeric_limits<T>::max();
  }
  return std / mean;
}

template <typename T>
void BoxCoxLambdaGuerrero(const T *x, int n, T *out, int period, T lower,
                          T upper) {
  if (n <= 2 * period) {
    *out = static_cast<T>(1.0);
    return;
  }
  for (int i = 0; i < n; ++i) {
    if (x[i] <= 0.0) {
      lower = std::max(lower, static_cast<T>(0.0));
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
  std::vector<T> logdata(n);
  std::transform(data, data + n, logdata.begin(),
                 [](T x) { return std::log(x); });
  double var;
  if (lambda == 0.0) {
    double mean = Mean(logdata.data(), n);
    var = Variance(logdata.data(), n, mean);
  } else {
    std::vector<T> transformed(n);
    std::transform(data, data + n, transformed.begin(), [lambda](T x) {
      return std::exp(lambda * std::log(x)) / lambda;
    });
    double mean = Mean(transformed.data(), n);
    var = Variance(transformed.data(), n, mean);
  }
  double sum_logdata = std::accumulate(logdata.begin(), logdata.end(), 0.0);
  return -static_cast<T>((lambda - 1) * sum_logdata - n / 2 * std::log(var));
}

template <typename T>
void BoxCoxLambdaLogLik(const T *x, int n, T *out, T lower, T upper) {
  T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.25);
  *out = Brent(BoxCoxLogLik<T>, lower, upper, tol, x, n);
}
} // namespace scalers
