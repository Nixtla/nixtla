#include "grouped_array_functions.h"

int GroupedArrayFloat32_Create(const float *data, indptr_t n_data,
                               indptr_t *indptr, indptr_t n_indptr,
                               int num_threads, GroupedArrayHandle *out) {
  *out = new GroupedArray<float>(data, n_data, indptr, n_indptr, num_threads);
  return 0;
}
int GroupedArrayFloat64_Create(const double *data, indptr_t n_data,
                               indptr_t *indptr, indptr_t n_indptr,
                               int num_threads, GroupedArrayHandle *out) {
  *out = new GroupedArray<double>(data, n_data, indptr, n_indptr, num_threads);
  return 0;
}

int GroupedArrayFloat32_Delete(GroupedArrayHandle handle) {
  delete reinterpret_cast<GroupedArray<float> *>(handle);
  return 0;
}
int GroupedArrayFloat64_Delete(GroupedArrayHandle handle) {
  delete reinterpret_cast<GroupedArray<double> *>(handle);
  return 0;
}
int GroupedArrayFloat32_TakeFromGroups(GroupedArrayHandle handle, int k,
                                       float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(TakeFromGroups<float>, 1, out, 0, k);
  return 0;
}
int GroupedArrayFloat64_TakeFromGroups(GroupedArrayHandle handle, int k,
                                       double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(TakeFromGroups<double>, 1, out, 0, k);
  return 0;
}
