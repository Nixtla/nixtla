#include "exponentially_weighted.h"

template <typename T>
inline void ExponentiallyWeightedMeanTransform(const T *data, int n, T *out,
                                               T alpha) {
  out[0] = data[0];
  for (int i = 1; i < n; ++i) {
    out[i] = alpha * data[i] + (1 - alpha) * out[i - 1];
  }
}

int GroupedArrayFloat32_ExponentiallyWeightedMeanTransform(
    GroupedArrayHandle handle, int lag, float alpha, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(ExponentiallyWeightedMeanTransform<float>, lag, out, alpha);
  return 0;
}
int GroupedArrayFloat64_ExponentiallyWeightedMeanTransform(
    GroupedArrayHandle handle, int lag, double alpha, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(ExponentiallyWeightedMeanTransform<double>, lag, out, alpha);
  return 0;
}
