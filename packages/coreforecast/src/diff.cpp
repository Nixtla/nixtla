#include "common.h"
#include "seasonal.h"

#include "diff.h"

template <typename T> int NumDiffs(const py::array_t<T> data, int max_d) {
  T out;
  diff::NumDiffs(data.data(), data.size(), &out, max_d);
  return static_cast<int>(out);
}

template <typename T>
int NumSeasDiffs(const py::array_t<T> data, int period, int max_d) {
  T out;
  diff::NumSeasDiffs(data.data(), data.size(), &out, period, max_d);
  return static_cast<int>(out);
}

template <typename T>
py::array_t<T> Difference(const py::array_t<T> data, int d) {
  py::array_t<T> out(data.size());
  seasonal::Difference(data.data(), data.size(), out.mutable_data(), d);
  return out;
}

template <typename T> void init_diffs_fns(py::module_ &m) {
  m.def("num_diffs", &NumDiffs<T>);
  m.def("num_seas_diffs", &NumSeasDiffs<T>);
  m.def("diff", &Difference<T>);
}

void init_diffs(py::module_ &m) {
  py::module_ diffs = m.def_submodule("differences");
  init_diffs_fns<float>(diffs);
  init_diffs_fns<double>(diffs);
}
