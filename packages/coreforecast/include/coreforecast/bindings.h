#pragma once

// Helpers for the pybind11 layer in src/. Kernels must not include this.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <span>

#include "common.h"

namespace py = pybind11;

// Contiguous input array. The kernels index a span over the buffer, so a
// strided view would be read as if it were contiguous and silently give results
// for the wrong elements.
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

// The whole buffer of a contiguous array as a span. The const overload is for
// inputs; only arrays this module allocated are viewed mutably.
template <typename T, int Flags>
inline std::span<const T> View(const py::array_t<T, Flags> &array) {
  return {array.data(), static_cast<size_t>(array.size())};
}
template <typename T, int Flags>
inline std::span<T> MutableView(py::array_t<T, Flags> &array) {
  return {array.mutable_data(), static_cast<size_t>(array.size())};
}

// The kernels assume NaNs only appear as a leading run, so the prefix is copied
// to the output and only the valid tail is handed to `f`, which is what
// GroupedArray::Transform already does for the grouped entry points.
template <typename T, typename Func>
inline py::array_t<T> SkipLeadingNaN(const py::array_t<T> &data_arg, Func f) {
  const auto data = AsContiguous(data_arg);
  py::array_t<T> out(data.size());
  const std::span<const T> in = View(data);
  const std::span<T> out_view = MutableView(out);
  const index_t start = FirstNotNaN(in, out_view);
  if (start < std::ssize(in)) {
    f(in.subspan(start), out_view.subspan(start));
  }
  return out;
}
