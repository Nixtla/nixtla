#include "common.h"

#include "exponentially_weighted.h"

template <typename T>
py::array_t<T> ExponentiallyWeightedMean(const py::array_t<T> data, T alpha,
                                         bool skipna = false) {
  return SkipLeadingNaN<T>(data, [&](const T *d, indptr_t n, T *o) {
    exponentially_weighted::MeanTransform<T>(d, n, o, alpha, skipna);
  });
}

void init_ew(py::module_ &m) {
  py::module_ ew = m.def_submodule("exponentially_weighted");
  ew.def("exponentially_weighted_mean", &ExponentiallyWeightedMean<double>,
         py::arg("data"), py::arg("alpha"), py::arg("skipna") = false);
  ew.def("exponentially_weighted_mean", &ExponentiallyWeightedMean<float>,
         py::arg("data"), py::arg("alpha"), py::arg("skipna") = false);
}
