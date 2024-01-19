#include "rolling.h"

template <typename T>
inline void RollingMeanTransform(const T *data, int n, T *out, int window_size,
                                 int min_samples) {
  T accum = static_cast<T>(0.0);
  int upper_limit = std::min(window_size, n);
  for (int i = 0; i < upper_limit; ++i) {
    accum += data[i];
    if (i + 1 < min_samples) {
      out[i] = std::numeric_limits<T>::quiet_NaN();
    } else {
      out[i] = accum / (i + 1);
    }
  }

  for (int i = window_size; i < n; ++i) {
    accum += data[i] - data[i - window_size];
    out[i] = accum / window_size;
  }
}

template <typename T>
inline void RollingStdTransform(const T *data, int n, T *out, int window_size,
                                int min_samples) {
  T tmp;
  RollingStdTransformWithStats(data, n, out, &tmp, false, window_size,
                               min_samples);
}

template <typename T> struct RollingMinTransform {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples) {
    RollingCompTransform(std::less<T>(), data, n, out, window_size,
                         min_samples);
  }
};

template <typename T> struct RollingMaxTransform {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples) const {
    RollingCompTransform(std::greater<T>(), data, n, out, window_size,
                         min_samples);
  }
};

template <typename Func, typename T, typename... Args>
inline void SeasonalRollingTransform(Func RollingTfm, const T *data, int n,
                                     T *out, int season_length, int window_size,
                                     int min_samples, Args &&...args) {
  int buff_size = n / season_length + (n % season_length > 0);
  T *season_data = new T[buff_size];
  T *season_out = new T[buff_size];
  std::fill_n(season_out, buff_size, std::numeric_limits<T>::quiet_NaN());
  for (int i = 0; i < season_length; ++i) {
    int season_n = n / season_length + (i < n % season_length);
    for (int j = 0; j < season_n; ++j) {
      season_data[j] = data[i + j * season_length];
    }
    RollingTfm(season_data, season_n, season_out, window_size, min_samples,
               std::forward<Args>(args)...);
    for (int j = 0; j < season_n; ++j) {
      out[i + j * season_length] = season_out[j];
    }
  }
  delete[] season_data;
  delete[] season_out;
}

template <typename T> struct SeasonalRollingMeanTransform {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingTransform(RollingMeanTransform<T>, data, n, out,
                             season_length, window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingStdTransform {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingTransform(RollingStdTransform<T>, data, n, out,
                             season_length, window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingMinTransform {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingTransform(RollingMinTransform<T>(), data, n, out,
                             season_length, window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingMaxTransform {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingTransform(RollingMaxTransform<T>(), data, n, out,
                             season_length, window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingQuantileTransform {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples, T p) {
    SeasonalRollingTransform(RollingQuantileTransform<T>, data, n, out,
                             season_length, window_size, min_samples, p);
  }
};

template <typename Func, typename T, typename... Args>
inline void RollingUpdate(Func RollingTfm, const T *data, int n, T *out,
                          int window_size, int min_samples, Args &&...args) {
  if (n < min_samples) {
    *out = std::numeric_limits<T>::quiet_NaN();
    return;
  }
  int n_samples = std::min(window_size, n);
  T *buffer = new T[n_samples];
  RollingTfm(data + n - n_samples, n_samples, buffer, window_size, min_samples,
             std::forward<Args>(args)...);
  *out = buffer[n_samples - 1];
  delete[] buffer;
}

template <typename T> struct RollingMeanUpdate {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples) {
    RollingUpdate(RollingMeanTransform<T>, data, n, out, window_size,
                  min_samples);
  }
};

template <typename T> struct RollingStdUpdate {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples) {
    RollingUpdate(RollingStdTransform<T>, data, n, out, window_size,
                  min_samples);
  }
};

template <typename T> struct RollingMinUpdate {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples) {
    RollingUpdate(RollingMinTransform<T>(), data, n, out, window_size,
                  min_samples);
  }
};

template <typename T> struct RollingMaxUpdate {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples) {
    RollingUpdate(RollingMaxTransform<T>(), data, n, out, window_size,
                  min_samples);
  }
};

template <typename T> struct RollingQuantileUpdate {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples, T p) {
    RollingUpdate(RollingQuantileTransform<T>, data, n, out, window_size,
                  min_samples, p);
  }
};

template <typename Func, typename T, typename... Args>
inline void SeasonalRollingUpdate(Func RollingUpdate, const T *data, int n,
                                  T *out, int season_length, int window_size,
                                  int min_samples, Args &&...args) {
  int season = n % season_length;
  int season_n = n / season_length + (season > 0);
  if (season_n < min_samples) {
    *out = std::numeric_limits<T>::quiet_NaN();
    return;
  }
  int n_samples = std::min(window_size, season_n);
  T *season_data = new T[n_samples];
  for (int i = 0; i < n_samples; ++i) {
    season_data[i] = data[n - 1 - (n_samples - 1 - i) * season_length];
  }
  RollingUpdate(season_data, n_samples, out, window_size, min_samples,
                std::forward<Args>(args)...);
  delete[] season_data;
}

template <typename T> struct SeasonalRollingMeanUpdate {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingUpdate(RollingMeanUpdate<T>(), data, n, out, season_length,
                          window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingStdUpdate {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingUpdate(RollingStdUpdate<T>(), data, n, out, season_length,
                          window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingMinUpdate {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingUpdate(RollingMinUpdate<T>(), data, n, out, season_length,
                          window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingMaxUpdate {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingUpdate(RollingMaxUpdate<T>(), data, n, out, season_length,
                          window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingQuantileUpdate {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples, T p) {
    SeasonalRollingUpdate(RollingQuantileUpdate<T>(), data, n, out,
                          season_length, window_size, min_samples, p);
  }
};

int GroupedArrayFloat32_RollingMeanTransform(GroupedArrayHandle handle, int lag,
                                             int window_size, int min_samples,
                                             float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(RollingMeanTransform<float>, lag, out, window_size,
                min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingMeanTransform(GroupedArrayHandle handle, int lag,
                                             int window_size, int min_samples,
                                             double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(RollingMeanTransform<double>, lag, out, window_size,
                min_samples);
  return 0;
}

int GroupedArrayFloat32_RollingStdTransform(GroupedArrayHandle handle, int lag,
                                            int window_size, int min_samples,
                                            float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(RollingStdTransform<float>, lag, out, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingStdTransform(GroupedArrayHandle handle, int lag,
                                            int window_size, int min_samples,
                                            double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(RollingStdTransform<double>, lag, out, window_size,
                min_samples);
  return 0;
}

int GroupedArrayFloat32_RollingMinTransform(GroupedArrayHandle handle, int lag,
                                            int window_size, int min_samples,
                                            float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(RollingMinTransform<float>(), lag, out, window_size,
                min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingMinTransform(GroupedArrayHandle handle, int lag,
                                            int window_size, int min_samples,
                                            double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(RollingMinTransform<double>(), lag, out, window_size,
                min_samples);
  return 0;
}

int GroupedArrayFloat32_RollingMaxTransform(GroupedArrayHandle handle, int lag,
                                            int window_size, int min_samples,
                                            float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(RollingMaxTransform<float>(), lag, out, window_size,
                min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingMaxTransform(GroupedArrayHandle handle, int lag,
                                            int window_size, int min_samples,
                                            double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(RollingMaxTransform<double>(), lag, out, window_size,
                min_samples);
  return 0;
}

int GroupedArrayFloat32_RollingQuantileTransform(GroupedArrayHandle handle,
                                                 int lag, float p,
                                                 int window_size,
                                                 int min_samples, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(RollingQuantileTransform<float>, lag, out, window_size,
                min_samples, p);
  return 0;
}
int GroupedArrayFloat64_RollingQuantileTransform(GroupedArrayHandle handle,
                                                 int lag, double p,
                                                 int window_size,
                                                 int min_samples, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(RollingQuantileTransform<double>, lag, out, window_size,
                min_samples, p);
  return 0;
}

int GroupedArrayFloat32_RollingMeanUpdate(GroupedArrayHandle handle, int lag,
                                          int window_size, int min_samples,
                                          float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RollingMeanUpdate<float>(), 1, out, lag, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingMeanUpdate(GroupedArrayHandle handle, int lag,
                                          int window_size, int min_samples,
                                          double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RollingMeanUpdate<double>(), 1, out, lag, window_size,
             min_samples);
  return 0;
}

int GroupedArrayFloat32_RollingStdUpdate(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RollingStdUpdate<float>(), 1, out, lag, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingStdUpdate(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RollingStdUpdate<double>(), 1, out, lag, window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_RollingMinUpdate(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RollingMinUpdate<float>(), 1, out, lag, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingMinUpdate(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RollingMinUpdate<double>(), 1, out, lag, window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_RollingMaxUpdate(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RollingMaxUpdate<float>(), 1, out, lag, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingMaxUpdate(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RollingMaxUpdate<double>(), 1, out, lag, window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_RollingQuantileUpdate(GroupedArrayHandle handle,
                                              int lag, float p, int window_size,
                                              int min_samples, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RollingQuantileUpdate<float>(), 1, out, lag, window_size,
             min_samples, p);
  return 0;
}
int GroupedArrayFloat64_RollingQuantileUpdate(GroupedArrayHandle handle,
                                              int lag, double p,
                                              int window_size, int min_samples,
                                              double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RollingQuantileUpdate<double>(), 1, out, lag, window_size,
             min_samples, p);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingMeanTransform(GroupedArrayHandle handle,
                                                     int lag, int season_length,
                                                     int window_size,
                                                     int min_samples,
                                                     float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(SeasonalRollingMeanTransform<float>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingMeanTransform(GroupedArrayHandle handle,
                                                     int lag, int season_length,
                                                     int window_size,
                                                     int min_samples,
                                                     double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(SeasonalRollingMeanTransform<double>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingStdTransform(GroupedArrayHandle handle,
                                                    int lag, int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(SeasonalRollingStdTransform<float>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingStdTransform(GroupedArrayHandle handle,
                                                    int lag, int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(SeasonalRollingStdTransform<double>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingMinTransform(GroupedArrayHandle handle,
                                                    int lag, int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(SeasonalRollingMinTransform<float>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingMinTransform(GroupedArrayHandle handle,
                                                    int lag, int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(SeasonalRollingMinTransform<double>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingMaxTransform(GroupedArrayHandle handle,
                                                    int lag, int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(SeasonalRollingMaxTransform<float>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingMaxTransform(GroupedArrayHandle handle,
                                                    int lag, int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(SeasonalRollingMaxTransform<double>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingQuantileTransform(
    GroupedArrayHandle handle, int lag, int season_length, float p,
    int window_size, int min_samples, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(SeasonalRollingQuantileTransform<float>(), lag, out,
                season_length, window_size, min_samples, p);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingQuantileTransform(
    GroupedArrayHandle handle, int lag, int season_length, double p,
    int window_size, int min_samples, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(SeasonalRollingQuantileTransform<double>(), lag, out,
                season_length, window_size, min_samples, p);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingMeanUpdate(GroupedArrayHandle handle,
                                                  int lag, int season_length,
                                                  int window_size,
                                                  int min_samples, float *out) {

  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(SeasonalRollingMeanUpdate<float>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingMeanUpdate(GroupedArrayHandle handle,
                                                  int lag, int season_length,
                                                  int window_size,
                                                  int min_samples,
                                                  double *out) {

  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(SeasonalRollingMeanUpdate<double>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingStdUpdate(GroupedArrayHandle handle,
                                                 int lag, int season_length,
                                                 int window_size,
                                                 int min_samples, float *out) {

  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(SeasonalRollingStdUpdate<float>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingStdUpdate(GroupedArrayHandle handle,
                                                 int lag, int season_length,
                                                 int window_size,
                                                 int min_samples, double *out) {

  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(SeasonalRollingStdUpdate<double>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingMinUpdate(GroupedArrayHandle handle,
                                                 int lag, int season_length,
                                                 int window_size,
                                                 int min_samples, float *out) {

  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(SeasonalRollingMinUpdate<float>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingMinUpdate(GroupedArrayHandle handle,
                                                 int lag, int season_length,
                                                 int window_size,
                                                 int min_samples, double *out) {

  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(SeasonalRollingMinUpdate<double>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingMaxUpdate(GroupedArrayHandle handle,
                                                 int lag, int season_length,
                                                 int window_size,
                                                 int min_samples, float *out) {

  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(SeasonalRollingMaxUpdate<float>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingMaxUpdate(GroupedArrayHandle handle,
                                                 int lag, int season_length,
                                                 int window_size,
                                                 int min_samples, double *out) {

  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(SeasonalRollingMaxUpdate<double>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}

int GroupedArrayFloat32_SeasonalRollingQuantileUpdate(
    GroupedArrayHandle handle, int lag, int season_length, float p,
    int window_size, int min_samples, float *out) {

  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(SeasonalRollingQuantileUpdate<float>(), 1, out, lag, season_length,
             window_size, min_samples, p);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingQuantileUpdate(
    GroupedArrayHandle handle, int lag, int season_length, double p,
    int window_size, int min_samples, double *out) {

  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(SeasonalRollingQuantileUpdate<double>(), 1, out, lag,
             season_length, window_size, min_samples, p);
  return 0;
}
