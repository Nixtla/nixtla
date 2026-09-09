#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <span>
#include <vector>

#include "doctest.h"

namespace helpers {

template <typename T> constexpr T NaN = std::numeric_limits<T>::quiet_NaN();

// Spans over test vectors, so kernel calls deduce T.
template <typename T> std::span<const T> In(const std::vector<T> &v) {
  return v;
}
template <typename T> std::span<T> Out(std::vector<T> &v) { return v; }
template <typename T> std::span<T> Out(T &scalar) { return {&scalar, 1}; }

// Fixed seed so a failure reproduces; uniform so no value is special.
template <typename T> std::vector<T> Random(int n, unsigned seed = 7) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);
  std::vector<T> out(n);
  for (auto &x : out) {
    x = static_cast<T>(dist(gen));
  }
  return out;
}

// Puts NaN at every position in `where`.
template <typename T>
std::vector<T> WithNaN(std::vector<T> x, std::initializer_list<int> where) {
  for (int i : where) {
    x[i] = NaN<T>;
  }
  return x;
}

// The values of data[lo, hi) that a NaN-aware statistic would use.
template <typename T>
std::vector<T> Valid(const std::vector<T> &data, int lo, int hi, bool skipna) {
  std::vector<T> out;
  for (int i = lo; i < hi; ++i) {
    if (!skipna || !std::isnan(data[i])) {
      out.push_back(data[i]);
    }
  }
  return out;
}

template <typename T> T Mean(const std::vector<T> &v) {
  double s = 0.0;
  for (auto x : v) {
    s += x;
  }
  return static_cast<T>(s / v.size());
}

// Sample standard deviation; NaN for fewer than two values, like the kernels.
template <typename T> T Std(const std::vector<T> &v) {
  if (v.size() < 2) {
    return NaN<T>;
  }
  double m = Mean(v);
  double s = 0.0;
  for (auto x : v) {
    s += (x - m) * (x - m);
  }
  return static_cast<T>(std::sqrt(s / (v.size() - 1)));
}

// numpy's default linear interpolation, on a sorted copy.
template <typename T> T Quantile(std::vector<T> v, T p) {
  std::sort(v.begin(), v.end());
  double pos = p * (v.size() - 1);
  auto i = static_cast<size_t>(pos);
  double g = pos - i;
  if (g == 0.0 || i + 1 == v.size()) {
    return v[i];
  }
  return static_cast<T>(v[i] + g * (v[i + 1] - v[i]));
}

// Element-wise comparison treating NaN == NaN, with a relative tolerance that
// suits the dtype.
template <typename T>
void CheckClose(const std::vector<T> &got, const std::vector<T> &want) {
  REQUIRE(got.size() == want.size());
  const double eps = std::is_same_v<T, float> ? 1e-4 : 1e-9;
  for (size_t i = 0; i < got.size(); ++i) {
    INFO("index ", i);
    if (std::isnan(want[i])) {
      CHECK(std::isnan(got[i]));
    } else {
      CHECK(got[i] == doctest::Approx(want[i]).epsilon(eps));
    }
  }
}

template <typename T> void CheckClose(T got, T want) {
  CheckClose(std::vector<T>{got}, std::vector<T>{want});
}

} // namespace helpers
