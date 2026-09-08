#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using indptr_t = int32_t;
namespace py = pybind11;

// Contiguous input array. The kernels take a raw pointer, so a strided view
// would be read as if it were contiguous and silently give results for the
// wrong elements.
template <typename T>
using CArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

// Ensures contiguity without putting c_style on the parameter itself: doing
// that makes array_t::check_ fail for a strided array, so pybind's
// no-conversion pass matches no overload and the conversion pass picks the
// first one registered, silently converting the input to that overload's dtype
// (float64, which is the one registered first).
template <typename T>
inline CArray<T> AsContiguous(const py::array_t<T> &data) {
  return CArray<T>::ensure(data);
}

// Counts that index into the buffers: zero or less makes the growing loops in
// the kernels start at -1 and read and write one element before them.
inline void RequirePositive(const char *name, int value) {
  if (value > 0) {
    return;
  }
  throw std::invalid_argument(std::string(name) + " must be greater than 0");
}

// Offsets into the buffers: a negative one reads and writes before their start.
inline void RequireNonNegative(const char *name, int value) {
  if (value < 0) {
    throw std::invalid_argument(std::string(name) + " must be non-negative");
  }
}

template <typename T> inline indptr_t FirstNotNaN(const T *data, indptr_t n) {
  indptr_t i = 0;
  while (i < n && std::isnan(data[i])) {
    ++i;
  }
  return i;
}

template <typename T>
inline indptr_t FirstNotNaN(const T *data, indptr_t n, T *out) {
  indptr_t i = 0;
  while (i < n && std::isnan(data[i])) {
    out[i++] = std::numeric_limits<T>::quiet_NaN();
  }
  return i;
}

// The kernels assume NaNs only appear as a leading run, so the prefix is copied
// to the output and only the valid tail is handed to `f`, which is what
// GroupedArray::Transform already does for the grouped entry points.
template <typename T, typename Func>
inline py::array_t<T> SkipLeadingNaN(const py::array_t<T> &data_arg, Func f) {
  const auto data = AsContiguous(data_arg);
  py::array_t<T> out(data.size());
  auto n = static_cast<indptr_t>(data.size());
  indptr_t start = FirstNotNaN(data.data(), n, out.mutable_data());
  if (start < n) {
    f(data.data() + start, n - start, out.mutable_data() + start);
  }
  return out;
}
