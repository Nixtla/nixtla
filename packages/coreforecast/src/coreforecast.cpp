#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_diffs(py::module_ &);
void init_exp(py::module_ &);
void init_ew(py::module_ &);
void init_ga(py::module_ &);
void init_roll(py::module_ &);
void init_sc(py::module_ &);
void init_seas(py::module_ &);

PYBIND11_MODULE(_lib, m) {
  init_diffs(m);
  init_exp(m);
  init_ew(m);
  init_ga(m);
  init_roll(m);
  init_sc(m);
  init_seas(m);
}
