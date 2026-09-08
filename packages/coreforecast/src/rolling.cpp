#include "common.h"

#include "rolling.h"

template <typename T, typename Func, typename... Args>
py::array_t<T> RollingOp(Func f, const py::array_t<T> data, int window_size,
                         int min_samples, Args... args) {
  return SkipLeadingNaN<T>(data, [&](const T *d, indptr_t n, T *o) {
    f(d, n, o, window_size, min_samples, std::forward<Args>(args)...);
  });
}

template <typename T>
py::array_t<T> RollingMean(const py::array_t<T> data, int window_size,
                           int min_samples, bool skipna = false) {
  return RollingOp(rolling::MeanTransform<T>, data, window_size, min_samples,
                   skipna);
}

template <typename T>
py::array_t<T> RollingStd(const py::array_t<T> data, int window_size,
                          int min_samples, bool skipna = false) {
  return RollingOp(rolling::StdTransform<T>, data, window_size, min_samples,
                   skipna);
}

template <typename T>
py::array_t<T> RollingMin(const py::array_t<T> data, int window_size,
                          int min_samples, bool skipna = false) {
  return RollingOp(rolling::MinTransform<T>, data, window_size, min_samples,
                   skipna);
}

template <typename T>
py::array_t<T> RollingMax(const py::array_t<T> data, int window_size,
                          int min_samples, bool skipna = false) {
  return RollingOp(rolling::MaxTransform<T>, data, window_size, min_samples,
                   skipna);
}

template <typename T>
py::array_t<T> RollingQuantile(const py::array_t<T> data, int window_size,
                               int min_samples, T p, bool skipna = false) {
  rolling::RequireProbability("p", p);
  return RollingOp(rolling::QuantileTransform<T>, data, window_size,
                   min_samples, p, skipna);
}

template <typename T, typename Func, typename... Args>
py::array_t<T> SeasonalRollingOp(Func f, const py::array_t<T> data,
                                 int season_length, int window_size,
                                 int min_samples, Args... args) {
  return SkipLeadingNaN<T>(data, [&](const T *d, indptr_t n, T *o) {
    f(d, n, o, season_length, window_size, min_samples,
      std::forward<Args>(args)...);
  });
}

template <typename T>
py::array_t<T> SeasonalRollingMean(const py::array_t<T> data, int season_length,
                                   int window_size, int min_samples,
                                   bool skipna = false) {
  return SeasonalRollingOp(rolling::SeasonalMeanTransform<T>, data,
                           season_length, window_size, min_samples, skipna);
}

template <typename T>
py::array_t<T> SeasonalRollingStd(const py::array_t<T> data, int season_length,
                                  int window_size, int min_samples,
                                  bool skipna = false) {
  return SeasonalRollingOp(rolling::SeasonalStdTransform<T>, data,
                           season_length, window_size, min_samples, skipna);
}

template <typename T>
py::array_t<T> SeasonalRollingMin(const py::array_t<T> data, int season_length,
                                  int window_size, int min_samples,
                                  bool skipna = false) {
  return SeasonalRollingOp(rolling::SeasonalMinTransform<T>, data,
                           season_length, window_size, min_samples, skipna);
}

template <typename T>
py::array_t<T> SeasonalRollingMax(const py::array_t<T> data, int season_length,
                                  int window_size, int min_samples,
                                  bool skipna = false) {
  return SeasonalRollingOp(rolling::SeasonalMaxTransform<T>, data,
                           season_length, window_size, min_samples, skipna);
}

template <typename T>
py::array_t<T> SeasonalRollingQuantile(const py::array_t<T> data,
                                       int season_length, int window_size,
                                       int min_samples, T p,
                                       bool skipna = false) {
  rolling::RequireProbability("p", p);
  return SeasonalRollingOp(rolling::SeasonalQuantileTransform<T>, data,
                           season_length, window_size, min_samples, p, skipna);
}

template <typename T> void init_roll_fns(py::module_ &m) {
  m.def("rolling_mean", &RollingMean<T>, py::arg("data"),
        py::arg("window_size"), py::arg("min_samples"),
        py::arg("skipna") = false);
  m.def("rolling_std", &RollingStd<T>, py::arg("data"), py::arg("window_size"),
        py::arg("min_samples"), py::arg("skipna") = false);
  m.def("rolling_min", &RollingMin<T>, py::arg("data"), py::arg("window_size"),
        py::arg("min_samples"), py::arg("skipna") = false);
  m.def("rolling_max", &RollingMax<T>, py::arg("data"), py::arg("window_size"),
        py::arg("min_samples"), py::arg("skipna") = false);
  m.def("rolling_quantile", &RollingQuantile<T>, py::arg("data"),
        py::arg("window_size"), py::arg("min_samples"), py::arg("p"),
        py::arg("skipna") = false);
  m.def("seasonal_rolling_mean", &SeasonalRollingMean<T>, py::arg("data"),
        py::arg("season_length"), py::arg("window_size"),
        py::arg("min_samples"), py::arg("skipna") = false);
  m.def("seasonal_rolling_std", &SeasonalRollingStd<T>, py::arg("data"),
        py::arg("season_length"), py::arg("window_size"),
        py::arg("min_samples"), py::arg("skipna") = false);
  m.def("seasonal_rolling_min", &SeasonalRollingMin<T>, py::arg("data"),
        py::arg("season_length"), py::arg("window_size"),
        py::arg("min_samples"), py::arg("skipna") = false);
  m.def("seasonal_rolling_max", &SeasonalRollingMax<T>, py::arg("data"),
        py::arg("season_length"), py::arg("window_size"),
        py::arg("min_samples"), py::arg("skipna") = false);
  m.def("seasonal_rolling_quantile", &SeasonalRollingQuantile<T>,
        py::arg("data"), py::arg("season_length"), py::arg("window_size"),
        py::arg("min_samples"), py::arg("p"), py::arg("skipna") = false);
}

void init_roll(py::module_ &m) {
  py::module_ roll = m.def_submodule("rolling");
  init_roll_fns<double>(roll);
  init_roll_fns<float>(roll);
}
