#pragma once

#include "export.h"
#include "grouped_array.h"

extern "C" {
DLL_EXPORT int
GroupedArrayFloat32_ExpandingMeanTransform(GroupedArrayHandle handle, int lag,
                                           float *out, float *agg);
DLL_EXPORT int
GroupedArrayFloat64_ExpandingMeanTransform(GroupedArrayHandle handle, int lag,
                                           double *out, double *agg);

DLL_EXPORT int
GroupedArrayFloat32_ExpandingStdTransform(GroupedArrayHandle handle, int lag,
                                          float *out, float *agg);
DLL_EXPORT int
GroupedArrayFloat64_ExpandingStdTransform(GroupedArrayHandle handle, int lag,
                                          double *out, double *agg);

DLL_EXPORT int
GroupedArrayFloat32_ExpandingMinTransform(GroupedArrayHandle handle, int lag,
                                          float *out);
DLL_EXPORT int
GroupedArrayFloat64_ExpandingMinTransform(GroupedArrayHandle handle, int lag,
                                          double *out);

DLL_EXPORT int
GroupedArrayFloat32_ExpandingMaxTransform(GroupedArrayHandle handle, int lag,
                                          float *out);
DLL_EXPORT int
GroupedArrayFloat64_ExpandingMaxTransform(GroupedArrayHandle handle, int lag,
                                          double *out);

DLL_EXPORT int
GroupedArrayFloat32_ExpandingQuantileTransform(GroupedArrayHandle handle,
                                               int lag, float p, float *out);
DLL_EXPORT int
GroupedArrayFloat64_ExpandingQuantileTransform(GroupedArrayHandle handle,
                                               int lag, double p, double *out);

DLL_EXPORT int
GroupedArrayFloat32_ExpandingQuantileUpdate(GroupedArrayHandle handle, int lag,
                                            float p, float *out);
DLL_EXPORT int
GroupedArrayFloat64_ExpandingQuantileUpdate(GroupedArrayHandle handle, int lag,
                                            double p, double *out);
}
