#pragma once

#include "export.h"
#include "grouped_array.h"

template <typename T> void Difference(const T *data, int n, T *out, int d) {
  if (d == 0) {
    std::copy(data, data + n, out);
    return;
  }
  std::fill(out, out + d, std::numeric_limits<T>::quiet_NaN());
  for (int i = d; i < n; ++i) {
    out[i] = data[i] - data[i - d];
  }
}

template <typename T> void Differences(const T *data, int n, int d, T *out) {
  Difference(data, n, out, d);
}

extern "C" {
DLL_EXPORT void Float32_Difference(const float *x, indptr_t n, int d,
                                   float *out);
DLL_EXPORT void Float64_Difference(const double *x, indptr_t n, int d,
                                   double *out);

DLL_EXPORT int Float32_NumDiffs(const float *x, indptr_t n, int max_d);
DLL_EXPORT int Float64_NumDiffs(const double *x, indptr_t n, int max_d);

DLL_EXPORT int Float32_NumSeasDiffs(const float *x, indptr_t n, int period,
                                    int max_d);
DLL_EXPORT int Float64_NumSeasDiffs(const double *x, indptr_t n, int period,
                                    int max_d);

DLL_EXPORT int Float32_Period(const float *x, size_t n, int period);
DLL_EXPORT int Float64_Period(const double *x, size_t n, int period);

DLL_EXPORT void GroupedArrayFloat32_NumDiffs(GroupedArrayHandle handle,
                                             int max_d, float *out);
DLL_EXPORT void GroupedArrayFloat64_NumDiffs(GroupedArrayHandle handle,
                                             int max_d, double *out);

DLL_EXPORT void GroupedArrayFloat32_NumSeasDiffs(GroupedArrayHandle handle,
                                                 int period, int max_d,
                                                 float *out);
DLL_EXPORT void GroupedArrayFloat64_NumSeasDiffs(GroupedArrayHandle handle,
                                                 int period, int max_d,
                                                 double *out);

DLL_EXPORT void
GroupedArrayFloat32_NumSeasDiffsPeriods(GroupedArrayHandle handle, int max_d,
                                        float *periods_and_out);
DLL_EXPORT void
GroupedArrayFloat64_NumSeasDiffsPeriods(GroupedArrayHandle handle, int max_d,
                                        double *periods_and_out);

DLL_EXPORT void GroupedArrayFloat32_Period(GroupedArrayHandle handle,
                                           size_t max_lag, float *out);
DLL_EXPORT void GroupedArrayFloat64_Period(GroupedArrayHandle handle,
                                           size_t max_lag, double *out);

DLL_EXPORT void GroupedArrayFloat32_Difference(GroupedArrayHandle handle, int d,
                                               float *out);
DLL_EXPORT void GroupedArrayFloat64_Difference(GroupedArrayHandle handle, int d,
                                               double *out);

DLL_EXPORT void GroupedArrayFloat32_Differences(GroupedArrayHandle handle,
                                                const indptr_t *ds, float *out);
DLL_EXPORT void GroupedArrayFloat64_Differences(GroupedArrayHandle handle,
                                                const indptr_t *ds,
                                                double *out);

DLL_EXPORT void GroupedArrayFloat32_InvertDifferences(
    GroupedArrayHandle handle, GroupedArrayHandle tails_handle,
    const indptr_t *out_indptr, float *out_data);
DLL_EXPORT void GroupedArrayFloat64_InvertDifferences(
    GroupedArrayHandle handle, GroupedArrayHandle tails_handle,
    const indptr_t *out_indptr, double *out_data);
}
