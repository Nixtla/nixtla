#include "common.h"

#include "seasonal.h"

template <typename T> int Period(const py::array_t<T> data, int max_lag) {
  T out;
  seasonal::GreatestAutocovariance(data.data(), data.size(), &out, max_lag);
  return static_cast<int>(out);
}

void init_seas(py::module_ &m) {
  py::module_ seas = m.def_submodule("seasonal");
  seas.def("period", &Period<float>);
  seas.def("period", &Period<double>);
}
