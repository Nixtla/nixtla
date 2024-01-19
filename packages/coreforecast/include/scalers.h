#pragma once

#include "export.h"
#include "grouped_array.h"

extern "C" {
DLL_EXPORT int GroupedArrayFloat32_MinMaxScalerStats(GroupedArrayHandle handle,
                                                     float *out);
DLL_EXPORT int GroupedArrayFloat64_MinMaxScalerStats(GroupedArrayHandle handle,
                                                     double *out);

DLL_EXPORT int
GroupedArrayFloat32_StandardScalerStats(GroupedArrayHandle handle, float *out);
DLL_EXPORT int
GroupedArrayFloat64_StandardScalerStats(GroupedArrayHandle handle, double *out);

DLL_EXPORT int
GroupedArrayFloat32_RobustIqrScalerStats(GroupedArrayHandle handle, float *out);
DLL_EXPORT int
GroupedArrayFloat64_RobustIqrScalerStats(GroupedArrayHandle handle,
                                         double *out);

DLL_EXPORT int
GroupedArrayFloat32_RobustMadScalerStats(GroupedArrayHandle handle, float *out);
DLL_EXPORT int
GroupedArrayFloat64_RobustMadScalerStats(GroupedArrayHandle handle,
                                         double *out);

DLL_EXPORT int GroupedArrayFloat32_ScalerTransform(GroupedArrayHandle handle,
                                                   const float *stats,
                                                   float *out);
DLL_EXPORT int GroupedArrayFloat64_ScalerTransform(GroupedArrayHandle handle,
                                                   const double *stats,
                                                   double *out);

DLL_EXPORT int
GroupedArrayFloat32_ScalerInverseTransform(GroupedArrayHandle handle,
                                           const float *stats, float *out);
DLL_EXPORT int
GroupedArrayFloat64_ScalerInverseTransform(GroupedArrayHandle handle,
                                           const double *stats, double *out);
}
