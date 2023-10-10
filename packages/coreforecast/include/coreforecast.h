#pragma once

#include <cstdint>

#ifdef _MSC_VER
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

typedef void *GroupedArrayHandle;

extern "C" {
DLL_EXPORT int GroupedArray_CreateFromArrays(float *data, int32_t n_data,
                                             int32_t *indptr, int32_t n_groups,
                                             GroupedArrayHandle *out);

DLL_EXPORT int GroupedArray_Delete(GroupedArrayHandle handle);

DLL_EXPORT int GroupedArray_MinMaxScalerStats(GroupedArrayHandle handle,
                                              double *out);

DLL_EXPORT int GroupedArray_StandardScalerStats(GroupedArrayHandle handle,
                                                double *out);

DLL_EXPORT int GroupedArray_RobustScalerIqrStats(GroupedArrayHandle handle,
                                                 double *out);

DLL_EXPORT int GroupedArray_RobustScalerMadStats(GroupedArrayHandle handle,
                                                 double *out);

DLL_EXPORT int GroupedArray_ScalerTransform(GroupedArrayHandle handle,
                                            double *stats, float *out);

DLL_EXPORT int GroupedArray_ScalerInverseTransform(GroupedArrayHandle handle,
                                                   double *stats, float *out);
}
