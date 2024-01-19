#pragma once

#include "export.h"
#include "grouped_array.h"

extern "C" {
DLL_EXPORT int GroupedArrayFloat32_ExponentiallyWeightedMeanTransform(
    GroupedArrayHandle handle, int lag, float alpha, float *out);
DLL_EXPORT int GroupedArrayFloat64_ExponentiallyWeightedMeanTransform(
    GroupedArrayHandle handle, int lag, double alpha, double *out);
}
