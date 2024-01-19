#include "export.h"
#include "grouped_array.h"

extern "C" {
DLL_EXPORT int GroupedArrayFloat32_Create(const float *data, indptr_t n_data,
                                          indptr_t *indptr, indptr_t n_indptr,
                                          int num_threads,
                                          GroupedArrayHandle *out);
DLL_EXPORT int GroupedArrayFloat64_Create(const double *data, indptr_t n_data,
                                          indptr_t *indptr, indptr_t n_indptr,
                                          int num_threads,
                                          GroupedArrayHandle *out);

DLL_EXPORT int GroupedArrayFloat32_Delete(GroupedArrayHandle handle);
DLL_EXPORT int GroupedArrayFloat64_Delete(GroupedArrayHandle handle);

DLL_EXPORT int GroupedArrayFloat32_TakeFromGroups(GroupedArrayHandle handle,
                                                  int k, float *out);
DLL_EXPORT int GroupedArrayFloat64_TakeFromGroups(GroupedArrayHandle handle,
                                                  int k, double *out);
}
