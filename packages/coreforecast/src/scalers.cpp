#include "scalers.h"
#include "brent.h"
#include "stats.h"

#include <algorithm>
#include <numeric>
#include <vector>

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
  T *buffer = new T[n];
  std::copy(data, data + n, buffer);
  T median = Quantile(buffer, static_cast<T>(0.5), n);
  T q1 = Quantile(buffer, static_cast<T>(0.25), n);
  T q3 = Quantile(buffer, static_cast<T>(0.75), n);
  stats[0] = median;
  stats[1] = q3 - q1;
  delete[] buffer;
}

template <typename T>
inline void RobustScalerMadStats(const T *data, int n, T *stats) {
  T *buffer = new T[n];
  std::copy(data, data + n, buffer);
  const T median = Quantile(buffer, static_cast<T>(0.5), n);
  for (int i = 0; i < n; ++i) {
    buffer[i] = std::abs(buffer[i] - median);
  }
  T mad = Quantile(buffer, static_cast<T>(0.5), n);
  stats[0] = median;
  stats[1] = mad;
  delete[] buffer;
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
void BoxCoxLambda_Guerrero(const T *x, int n, T *out, int period, T lower,
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

template <typename T> T BoxCox_LogLik(T lambda, const T *data, int n) {
  T *logdata = new T[n];
  std::transform(data, data + n, logdata, [](T x) { return std::log(x); });
  double var;
  if (lambda == 0.0) {
    double mean = Mean(logdata, n);
    var = Variance(logdata, n, mean);
  } else {
    T *transformed = new T[n];
    std::transform(data, data + n, transformed, [lambda](T x) {
      return std::exp(lambda * std::log(x)) / lambda;
    });
    double mean = Mean(transformed, n);
    var = Variance(transformed, n, mean);
    delete[] transformed;
  }
  double sum_logdata = std::accumulate(logdata, logdata + n, 0.0);
  delete[] logdata;
  return -static_cast<T>((lambda - 1) * sum_logdata - n / 2 * std::log(var));
}

template <typename T>
void BoxCoxLambda_LogLik(const T *x, int n, T *out, T lower, T upper) {
  T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.25);
  *out = Brent(BoxCox_LogLik<T>, lower, upper, tol, x, n);
}

float Float32_BoxCoxLambdaGuerrero(const float *x, int n, int period,
                                   float lower, float upper) {
  float out;
  BoxCoxLambda_Guerrero<float>(x, n, &out, period, lower, upper);
  return out;
}
double Float64_BoxCoxLambdaGuerrero(const double *x, int n, int period,
                                    double lower, double upper) {
  double out;
  BoxCoxLambda_Guerrero<double>(x, n, &out, period, lower, upper);
  return out;
}

float Float32_BoxCoxLambdaLogLik(const float *x, int n, float lower,
                                 float upper) {
  float out;
  BoxCoxLambda_LogLik<float>(x, n, &out, lower, upper);
  return out;
}
double Float64_BoxCoxLambdaLogLik(const double *x, int n, double lower,
                                  double upper) {
  double out;
  BoxCoxLambda_LogLik<double>(x, n, &out, lower, upper);
  return out;
}

void Float32_BoxCoxTransform(const float *x, int n, float lambda, float *out) {
  std::transform(x, x + n, out, [lambda](float x) {
    return BoxCoxTransform<float>(x, lambda, 0.0);
  });
}
void Float64_BoxCoxTransform(const double *x, int n, double lambda,
                             double *out) {
  std::transform(x, x + n, out, [lambda](double x) {
    return BoxCoxTransform<double>(x, lambda, 0.0);
  });
}

void Float32_BoxCoxInverseTransform(const float *x, int n, float lambda,
                                    float *out) {
  std::transform(x, x + n, out, [lambda](float x) {
    return BoxCoxInverseTransform<float>(x, lambda, 0.0);
  });
}
void Float64_BoxCoxInverseTransform(const double *x, int n, double lambda,
                                    double *out) {
  std::transform(x, x + n, out, [lambda](double x) {
    return BoxCoxInverseTransform<double>(x, lambda, 0.0);
  });
}

int GroupedArrayFloat32_MinMaxScalerStats(GroupedArrayHandle handle,
                                          float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(MinMaxScalerStats<float>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat64_MinMaxScalerStats(GroupedArrayHandle handle,
                                          double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(MinMaxScalerStats<double>, 2, out, 0);
  return 0;
}

int GroupedArrayFloat32_StandardScalerStats(GroupedArrayHandle handle,
                                            float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(StandardScalerStats<float>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat64_StandardScalerStats(GroupedArrayHandle handle,
                                            double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(StandardScalerStats<double>, 2, out, 0);
  return 0;
}

int GroupedArrayFloat32_RobustIqrScalerStats(GroupedArrayHandle handle,
                                             float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RobustScalerIqrStats<float>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat64_RobustIqrScalerStats(GroupedArrayHandle handle,
                                             double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RobustScalerIqrStats<double>, 2, out, 0);
  return 0;
}

int GroupedArrayFloat32_RobustMadScalerStats(GroupedArrayHandle handle,
                                             float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RobustScalerMadStats<float>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat64_RobustMadScalerStats(GroupedArrayHandle handle,
                                             double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RobustScalerMadStats<double>, 2, out, 0);
  return 0;
}

int GroupedArrayFloat32_ScalerTransform(GroupedArrayHandle handle,
                                        const float *stats, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->ScalerTransform(CommonScalerTransform<float>, stats, out);
  return 0;
}
int GroupedArrayFloat64_ScalerTransform(GroupedArrayHandle handle,
                                        const double *stats, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->ScalerTransform(CommonScalerTransform<double>, stats, out);
  return 0;
}

int GroupedArrayFloat32_ScalerInverseTransform(GroupedArrayHandle handle,
                                               const float *stats, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->ScalerTransform(CommonScalerInverseTransform<float>, stats, out);
  return 0;
}
int GroupedArrayFloat64_ScalerInverseTransform(GroupedArrayHandle handle,
                                               const double *stats,
                                               double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->ScalerTransform(CommonScalerInverseTransform<double>, stats, out);
  return 0;
}

int GroupedArrayFloat32_BoxCoxLambdaGuerrero(GroupedArrayHandle handle,
                                             int period, float lower,
                                             float upper, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(BoxCoxLambda_Guerrero<float>, 2, out, 0, period, lower, upper);
  return 0;
}
int GroupedArrayFloat64_BoxCoxLambdaGuerrero(GroupedArrayHandle handle,
                                             int period, double lower,
                                             double upper, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(BoxCoxLambda_Guerrero<double>, 2, out, 0, period, lower, upper);
  return 0;
}

void GroupedArrayFloat32_BoxCoxLambdaLogLik(GroupedArrayHandle handle,
                                            float lower, float upper,
                                            float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(BoxCoxLambda_LogLik<float>, 2, out, 0, lower, upper);
}
void GroupedArrayFloat64_BoxCoxLambdaLogLik(GroupedArrayHandle handle,
                                            double lower, double upper,
                                            double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(BoxCoxLambda_LogLik<double>, 2, out, 0, lower, upper);
}

int GroupedArrayFloat32_BoxCoxTransform(GroupedArrayHandle handle,
                                        const float *lambdas, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->ScalerTransform(BoxCoxTransform<float>, lambdas, out);
  return 0;
}
int GroupedArrayFloat64_BoxCoxTransform(GroupedArrayHandle handle,
                                        const double *lambdas, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->ScalerTransform(BoxCoxTransform<double>, lambdas, out);
  return 0;
}

int GroupedArrayFloat32_BoxCoxInverseTransform(GroupedArrayHandle handle,
                                               const float *lambdas,
                                               float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->ScalerTransform(BoxCoxInverseTransform<float>, lambdas, out);
  return 0;
}
int GroupedArrayFloat64_BoxCoxInverseTransform(GroupedArrayHandle handle,
                                               const double *lambdas,
                                               double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->ScalerTransform(BoxCoxInverseTransform<double>, lambdas, out);
  return 0;
}
