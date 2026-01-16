#include "common.h"

#include "exponentially_weighted.h"

template <typename T>
py::array_t<T> ExponentiallyWeightedMean(
    const py::array_t<T, py::array::c_style | py::array::forcecast> data,
    T alpha, bool skipna = false) {
  py::array_t<T> out(data.size());
  exponentially_weighted::MeanTransform<T>(data.data(), data.size(),
                                           out.mutable_data(), alpha, skipna);
  return out;
}

void init_ew(py::module_ &m) {
  py::module_ ew = m.def_submodule("exponentially_weighted");
  ew.def("exponentially_weighted_mean", &ExponentiallyWeightedMean<float>,
         py::arg("data"), py::arg("alpha"), py::arg("skipna") = false);
  ew.def("exponentially_weighted_mean", &ExponentiallyWeightedMean<double>,
         py::arg("data"), py::arg("alpha"), py::arg("skipna") = false);
}
