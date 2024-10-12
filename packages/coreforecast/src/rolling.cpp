#include "common.h"

#include "rolling.h"

template <typename T, typename Func, typename... Args>
py::array_t<T> RollingOp(Func f, const py::array_t<T> data, int window_size,
                         int min_samples, Args... args) {
  py::array_t<T> out(data.size());
  f(data.data(), data.size(), out.mutable_data(), window_size, min_samples,
    std::forward<Args>(args)...);
  return out;
}

template <typename T>
py::array_t<T> RollingMean(const py::array_t<T> data, int window_size,
                           int min_samples) {
  return RollingOp(rolling::MeanTransform<T>, data, window_size, min_samples);
}

template <typename T>
py::array_t<T> RollingStd(const py::array_t<T> data, int window_size,
                          int min_samples) {
  return RollingOp(rolling::StdTransform<T>, data, window_size, min_samples);
}

template <typename T>
py::array_t<T> RollingMin(const py::array_t<T> data, int window_size,
                          int min_samples) {
  return RollingOp(rolling::MinTransform<T>, data, window_size, min_samples);
}

template <typename T>
py::array_t<T> RollingMax(const py::array_t<T> data, int window_size,
                          int min_samples) {
  return RollingOp(rolling::MaxTransform<T>, data, window_size, min_samples);
}

template <typename T>
py::array_t<T> RollingQuantile(const py::array_t<T> data, int window_size,
                               int min_samples, T p) {
  return RollingOp(rolling::QuantileTransform<T>, data, window_size,
                   min_samples, p);
}

template <typename T, typename Func, typename... Args>
py::array_t<T> SeasonalRollingOp(Func f, const py::array_t<T> data,
                                 int season_length, int window_size,
                                 int min_samples, Args... args) {
  py::array_t<T> out(data.size());
  f(data.data(), data.size(), out.mutable_data(), season_length, window_size,
    min_samples, std::forward<Args>(args)...);
  return out;
}

template <typename T>
py::array_t<T> SeasonalRollingMean(const py::array_t<T> data, int season_length,
                                   int window_size, int min_samples) {
  return SeasonalRollingOp(rolling::SeasonalMeanTransform<T>, data,
                           season_length, window_size, min_samples);
}

template <typename T>
py::array_t<T> SeasonalRollingStd(const py::array_t<T> data, int season_length,
                                  int window_size, int min_samples) {
  return SeasonalRollingOp(rolling::SeasonalStdTransform<T>, data,
                           season_length, window_size, min_samples);
}

template <typename T>
py::array_t<T> SeasonalRollingMin(const py::array_t<T> data, int season_length,
                                  int window_size, int min_samples) {
  return SeasonalRollingOp(rolling::SeasonalMinTransform<T>, data,
                           season_length, window_size, min_samples);
}

template <typename T>
py::array_t<T> SeasonalRollingMax(const py::array_t<T> data, int season_length,
                                  int window_size, int min_samples) {
  return SeasonalRollingOp(rolling::SeasonalMaxTransform<T>, data,
                           season_length, window_size, min_samples);
}

template <typename T>
py::array_t<T> SeasonalRollingQuantile(const py::array_t<T> data,
                                       int season_length, int window_size,
                                       int min_samples, T p) {
  return SeasonalRollingOp(rolling::SeasonalQuantileTransform<T>, data,
                           season_length, window_size, min_samples, p);
}

template <typename T> void init_roll_fns(py::module_ &m) {
  m.def("rolling_mean", &RollingMean<T>);
  m.def("rolling_std", &RollingStd<T>);
  m.def("rolling_min", &RollingMin<T>);
  m.def("rolling_max", &RollingMax<T>);
  m.def("rolling_quantile", &RollingQuantile<T>);
  m.def("seasonal_rolling_mean", &SeasonalRollingMean<T>);
  m.def("seasonal_rolling_std", &SeasonalRollingStd<T>);
  m.def("seasonal_rolling_min", &SeasonalRollingMin<T>);
  m.def("seasonal_rolling_max", &SeasonalRollingMax<T>);
  m.def("seasonal_rolling_quantile", &SeasonalRollingQuantile<T>);
}

void init_roll(py::module_ &m) {
  py::module_ roll = m.def_submodule("rolling");
  init_roll_fns<float>(roll);
  init_roll_fns<double>(roll);
}
