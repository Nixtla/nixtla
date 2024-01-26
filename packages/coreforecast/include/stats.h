#pragma once

#include <algorithm>
#include <numeric>

template <typename T> inline T Quantile(T *data, T p, int n) {
  T i_plus_g = p * (n - 1);
  int i = static_cast<int>(i_plus_g);
  T g = i_plus_g - i;
  std::nth_element(data, data + i, data + n);
  T out = data[i];
  if (g > 0.0) {
    std::nth_element(data, data + i + 1, data + n);
    out += g * (data[i + 1] - out);
  }
  return out;
}

template <typename T> inline T SortedQuantile(T *data, T p, int n) {
  T i_plus_g = p * (n - 1);
  int i = static_cast<int>(i_plus_g);
  T g = i_plus_g - i;
  T out = data[i];
  if (g > 0.0) {
    out += g * (data[i + 1] - out);
  }
  return out;
}

template <typename T> double Mean(const T *data, int n) {
  double sum = std::accumulate(data, data + n, 0.0);
  return sum / n;
}

template <typename T>
double StandardDeviation(const T *data, int n, double mean, int ddof = 0) {
  if (n <= ddof) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double sum_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    sum_sq += (data[i] - mean) * (data[i] - mean);
  }
  return std::sqrt(sum_sq / (n - ddof));
}
