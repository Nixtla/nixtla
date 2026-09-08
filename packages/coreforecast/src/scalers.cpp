#include "common.h"

#include "scalers.h"

template <typename T>
T BoxCoxLambdaGuerrero(py::array_t<T> data, int period, T lower, T upper) {
  T out;
  const auto x = AsContiguous(data);
  scalers::BoxCoxLambdaGuerrero(x.data(), x.size(), &out, period, lower, upper);
  return out;
}

template <typename T>
T BoxCoxLambdaLogLik(py::array_t<T> data, T lower, T upper) {
  T out;
  const auto x = AsContiguous(data);
  scalers::BoxCoxLambdaLogLik(x.data(), x.size(), &out, lower, upper);
  return out;
}

template <typename T>
py::array_t<T> BoxCoxTransform(py::array_t<T> data, T lambda) {
  const auto in = AsContiguous(data);
  py::array_t<T> out(in.size());
  std::transform(
      in.data(), in.data() + in.size(), out.mutable_data(),
      [lambda](T x) { return scalers::BoxCoxTransform<T>(x, lambda, 0.0); });
  return out;
}

template <typename T>
py::array_t<T> BoxCoxInverseTransform(const py::array_t<T> data, T lambda) {
  const auto in = AsContiguous(data);
  py::array_t<T> out(in.size());
  std::transform(in.data(), in.data() + in.size(), out.mutable_data(),
                 [lambda](T x) {
                   return scalers::BoxCoxInverseTransform<T>(x, lambda, 0.0);
                 });
  return out;
}

template <typename T> void init_sc_fns(py::module_ &m) {
  m.def("boxcox_lambda_guerrero", &BoxCoxLambdaGuerrero<T>);
  m.def("boxcox_lambda_loglik", &BoxCoxLambdaLogLik<T>);
  m.def("boxcox", &BoxCoxTransform<T>);
  m.def("inv_boxcox", &BoxCoxInverseTransform<T>);
}

void init_sc(py::module_ &m) {
  py::module_ sc = m.def_submodule("scalers");
  init_sc_fns<double>(sc);
  init_sc_fns<float>(sc);
}
