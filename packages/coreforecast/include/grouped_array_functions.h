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

DLL_EXPORT int GroupedArrayFloat32_IndexFromEnd(GroupedArrayHandle handle,
                                                int k, float *out);
DLL_EXPORT int GroupedArrayFloat64_IndexFromEnd(GroupedArrayHandle handle,
                                                int k, double *out);

DLL_EXPORT void GroupedArrayFloat32_Head(GroupedArrayHandle handle, int k,
                                         float *out);
DLL_EXPORT void GroupedArrayFloat64_Head(GroupedArrayHandle handle, int k,
                                         double *out);

DLL_EXPORT void GroupedArrayFloat32_Tail(GroupedArrayHandle handle, int k,
                                         float *out);
DLL_EXPORT void GroupedArrayFloat64_Tail(GroupedArrayHandle handle, int k,
                                         double *out);

DLL_EXPORT void GroupedArrayFloat32_Tails(GroupedArrayHandle handle,
                                          const indptr_t *indptr_out,
                                          float *out);
DLL_EXPORT void GroupedArrayFloat64_Tails(GroupedArrayHandle handle,
                                          const indptr_t *indptr_out,
                                          double *out);

DLL_EXPORT void GroupedArrayFloat32_Append(GroupedArrayHandle handle,
                                           GroupedArrayHandle other_handle,
                                           const indptr_t *out_indptr,
                                           float *out_data);
DLL_EXPORT void GroupedArrayFloat64_Append(GroupedArrayHandle handle,
                                           GroupedArrayHandle other_handle,
                                           const indptr_t *out_indptr,
                                           double *out_data);
}
