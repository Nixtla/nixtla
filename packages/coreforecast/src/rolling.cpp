#include "common.h"

#include "rolling.h"

// Non-inline dispatch functions to ensure correct template selection
template <typename T>
void MeanTransformDispatch(const T *data, int n, T *out, int window_size,
                           int min_samples, bool skipna) {
  if (skipna) {
    rolling::Transform<T, rolling::MeanAccumulator<T, true>>(
        data, n, out, window_size, min_samples);
  } else {
    rolling::Transform<T, rolling::MeanAccumulator<T, false>>(
        data, n, out, window_size, min_samples);
  }
}

template <typename T>
void MinTransformDispatch(const T *data, int n, T *out, int window_size,
                          int min_samples, bool skipna) {
  if (skipna) {
    rolling::Transform<
        T, rolling::CompAccumulator<T, std::greater_equal<T>, true>>(
        data, n, out, window_size, min_samples);
  } else {
    rolling::Transform<
        T, rolling::CompAccumulator<T, std::greater_equal<T>, false>>(
        data, n, out, window_size, min_samples);
  }
}

template <typename T>
void MaxTransformDispatch(const T *data, int n, T *out, int window_size,
                          int min_samples, bool skipna) {
  if (skipna) {
    rolling::Transform<T,
                       rolling::CompAccumulator<T, std::less_equal<T>, true>>(
        data, n, out, window_size, min_samples);
  } else {
    rolling::Transform<T,
                       rolling::CompAccumulator<T, std::less_equal<T>, false>>(
        data, n, out, window_size, min_samples);
  }
}

template <typename T>
void QuantileTransformDispatch(const T *data, int n, T *out, int window_size,
                               int min_samples, T p, bool skipna) {
  if (skipna) {
    rolling::Transform<T, rolling::QuantileAccumulator<T, true>>(
        data, n, out, window_size, min_samples, p);
  } else {
    rolling::Transform<T, rolling::QuantileAccumulator<T, false>>(
        data, n, out, window_size, min_samples, p);
  }
}

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
                           int min_samples, bool skipna = false) {
  return RollingOp(MeanTransformDispatch<T>, data, window_size, min_samples,
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
  return RollingOp(MinTransformDispatch<T>, data, window_size, min_samples,
                   skipna);
}

template <typename T>
py::array_t<T> RollingMax(const py::array_t<T> data, int window_size,
                          int min_samples, bool skipna = false) {
  return RollingOp(MaxTransformDispatch<T>, data, window_size, min_samples,
                   skipna);
}

template <typename T>
py::array_t<T> RollingQuantile(const py::array_t<T> data, int window_size,
                               int min_samples, T p, bool skipna = false) {
  py::array_t<T> out(data.size());
  QuantileTransformDispatch<T>(data.data(), data.size(), out.mutable_data(),
                               window_size, min_samples, p, skipna);
  return out;
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
  init_roll_fns<float>(roll);
  init_roll_fns<double>(roll);
}

// Explicit template instantiations to ensure both SkipNA specializations are
// compiled
namespace rolling {
template class MeanAccumulator<float, true>;
template class MeanAccumulator<float, false>;
template class MeanAccumulator<double, true>;
template class MeanAccumulator<double, false>;

template class CompAccumulator<float, std::greater_equal<float>, true>;
template class CompAccumulator<float, std::greater_equal<float>, false>;
template class CompAccumulator<float, std::less_equal<float>, true>;
template class CompAccumulator<float, std::less_equal<float>, false>;
template class CompAccumulator<double, std::greater_equal<double>, true>;
template class CompAccumulator<double, std::greater_equal<double>, false>;
template class CompAccumulator<double, std::less_equal<double>, true>;
template class CompAccumulator<double, std::less_equal<double>, false>;

template class QuantileAccumulator<float, true>;
template class QuantileAccumulator<float, false>;
template class QuantileAccumulator<double, true>;
template class QuantileAccumulator<double, false>;

// Explicit instantiations of Transform functions
template void MeanTransform<float>(const float *, int, float *, int, int, bool);
template void MeanTransform<double>(const double *, int, double *, int, int,
                                    bool);
template void MinTransform<float>(const float *, int, float *, int, int, bool);
template void MinTransform<double>(const double *, int, double *, int, int,
                                   bool);
template void MaxTransform<float>(const float *, int, float *, int, int, bool);
template void MaxTransform<double>(const double *, int, double *, int, int,
                                   bool);
template void QuantileTransform<float>(const float *, int, float *, int, int,
                                       float, bool);
template void QuantileTransform<double>(const double *, int, double *, int, int,
                                        double, bool);
} // namespace rolling
