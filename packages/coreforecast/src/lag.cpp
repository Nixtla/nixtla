#include "lag.h"

template <typename T> inline void LagTransform(const T *data, int n, T *out) {
  std::copy(data, data + n, out);
}

int GroupedArrayFloat32_LagTransform(GroupedArrayHandle handle, int lag,
                                     float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(LagTransform<float>, lag, out);
  return 0;
}
int GroupedArrayFloat64_LagTransform(GroupedArrayHandle handle, int lag,
                                     double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(LagTransform<double>, lag, out);
  return 0;
}
