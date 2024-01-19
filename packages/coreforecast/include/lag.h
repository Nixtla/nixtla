#pragma once

#include "export.h"
#include "grouped_array.h"

extern "C" {
DLL_EXPORT int GroupedArrayFloat32_LagTransform(GroupedArrayHandle handle,
                                                int lag, float *out);
DLL_EXPORT int GroupedArrayFloat64_LagTransform(GroupedArrayHandle handle,
                                                int lag, double *out);
}
