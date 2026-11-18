#pragma once

// Shared by every kernel. Nothing here may depend on Python or pybind11: the
// kernels are compiled and tested on their own; the binding-side helpers live
// in bindings.h.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

// The one type for lengths, offsets and counts, including the GroupedArray
// indptr as it is stored. Signed so that mixing it with the int parameters
// that come from Python never changes the sign of an expression, and 64-bit
// so neither a group nor the whole array is limited to 2^31 elements.
using index_t = std::int64_t;

// Counts that index into the buffers: zero or less makes the growing loops in
// the kernels start at -1 and read and write one element before them.
inline void RequirePositive(const char *name, index_t value) {
  if (value > 0) {
    return;
  }
  throw std::invalid_argument(std::string(name) + " must be greater than 0");
}

// Offsets into the buffers: a negative one reads and writes before their start.
// Checked in the value's own type: a floating-point NaN (a period from a group
// that had no valid data) compares false and travels on to the driver, which
// skips that group; casting it to an integer first would be undefined.
template <typename V>
inline void RequireNonNegative(const char *name, V value) {
  if (value < 0) {
    throw std::invalid_argument(std::string(name) + " must be non-negative");
  }
}

// Quantile levels index into the sorted window, so a value outside [0, 1]
// indexes past its end. NaN fails both comparisons and is rejected too.
inline void RequireProbability(const char *name, double value) {
  if (value >= 0.0 && value <= 1.0) {
    return;
  }
  throw std::invalid_argument(std::string(name) + " must be between 0 and 1");
}

// Parameters of the rolling kernels. Built through Checked() at the Python
// boundary, once per call, before any output is allocated or thread spawned;
// the kernels trust what they receive and only assert it in debug builds.
struct Window {
  index_t window_size;
  index_t min_samples;
  bool skipna;

  static Window Checked(index_t window_size, index_t min_samples, bool skipna) {
    RequirePositive("window_size", window_size);
    RequirePositive("min_samples", min_samples);
    return {window_size, min_samples, skipna};
  }
};

struct SeasonalWindow {
  index_t season_length;
  Window window;

  static SeasonalWindow Checked(index_t season_length, index_t window_size,
                                index_t min_samples, bool skipna) {
    RequirePositive("season_length", season_length);
    return {season_length, Window::Checked(window_size, min_samples, skipna)};
  }
};

template <typename T> constexpr T kNaN = std::numeric_limits<T>::quiet_NaN();

template <typename T> inline void FillNaN(std::span<T> out) {
  std::fill(out.begin(), out.end(), kNaN<T>);
}

template <typename T> inline index_t FirstNotNaN(std::span<const T> data) {
  index_t i = 0;
  const index_t n = std::ssize(data);
  while (i < n && std::isnan(data[i])) {
    ++i;
  }
  return i;
}

// Also writes NaN over the prefix it skips.
template <typename T>
inline index_t FirstNotNaN(std::span<const T> data, std::span<T> out) {
  index_t i = 0;
  const index_t n = std::ssize(data);
  while (i < n && std::isnan(data[i])) {
    out[i++] = kNaN<T>;
  }
  return i;
}
