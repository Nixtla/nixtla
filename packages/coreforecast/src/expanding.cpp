#include "common.h"
#include "rolling.h"

#include "expanding.h"

template <typename T, typename Func, typename... Args>
py::array_t<T> ExpandingOp(Func f, const py::array_t<T> data, Args... args) {
  py::array_t<T> out(data.size());
  f(data.data(), data.size(), out.mutable_data(), std::forward<Args>(args)...);
  return out;
}

template <typename T>
py::array_t<T> ExpandingMean(const py::array_t<T> data, bool skipna = false) {
  T tmp;
  return ExpandingOp(expanding::MeanTransform<T>, data, &tmp, skipna);
}

template <typename T>
py::array_t<T> ExpandingStd(const py::array_t<T> data, bool skipna = false) {
  T tmp[3];
  return ExpandingOp(expanding::StdTransform<T>, data, tmp, skipna);
}

template <typename T>
py::array_t<T> ExpandingMin(const py::array_t<T> data, bool skipna = false) {
  return ExpandingOp(expanding::MinTransform<T>, data, skipna);
}

template <typename T>
py::array_t<T> ExpandingMax(const py::array_t<T> data, bool skipna = false) {
  return ExpandingOp(expanding::MaxTransform<T>, data, skipna);
}

template <typename T>
py::array_t<T> ExpandingQuantile(const py::array_t<T> data, T p,
                                 bool skipna = false) {
  return ExpandingOp(expanding::QuantileTransform<T>, data, p, skipna);
}

template <typename T> void init_exp_fns(py::module_ &m) {
  m.def("expanding_mean", &ExpandingMean<T>, py::arg("data"),
        py::arg("skipna") = false);
  m.def("expanding_std", &ExpandingStd<T>, py::arg("data"),
        py::arg("skipna") = false);
  m.def("expanding_min", &ExpandingMin<T>, py::arg("data"),
        py::arg("skipna") = false);
  m.def("expanding_max", &ExpandingMax<T>, py::arg("data"),
        py::arg("skipna") = false);
  m.def("expanding_quantile", &ExpandingQuantile<T>, py::arg("data"),
        py::arg("p"), py::arg("skipna") = false);
}

void init_exp(py::module_ &m) {
  py::module_ exp = m.def_submodule("expanding");
  init_exp_fns<float>(exp);
  init_exp_fns<double>(exp);
}
