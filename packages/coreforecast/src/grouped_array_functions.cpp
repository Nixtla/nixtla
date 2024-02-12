#include "grouped_array_functions.h"

template <typename T>
inline void IndexFromEnd(const T *data, int n, T *out, int k) {
  if (k > n) {
    *out = std::numeric_limits<T>::quiet_NaN();
  } else {
    *out = data[n - 1 - k];
  }
}

template <typename T> inline void Head(const T *data, int n, T *out, int k) {
  int m = std::min(k, n);
  std::copy(data, data + m, out);
  std::fill(out + m, out + k, std::numeric_limits<T>::quiet_NaN());
}

template <typename T> inline void Tail(const T *data, int n, T *out, int k) {
  int m = std::min(k, n);
  std::fill(out, out + k - m, std::numeric_limits<T>::quiet_NaN());
  std::copy(data + n - m, data + n, out + k - m);
}

template <typename T>
inline void Append(const T *data, int n, const T *other_data, int other_n,
                   T *out) {
  std::copy(data, data + n, out);
  std::copy(other_data, other_data + other_n, out + n);
}

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

int GroupedArrayFloat32_IndexFromEnd(GroupedArrayHandle handle, int k,
                                     float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(IndexFromEnd<float>, 1, out, 0, k);
  return 0;
}
int GroupedArrayFloat64_IndexFromEnd(GroupedArrayHandle handle, int k,
                                     double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(IndexFromEnd<double>, 1, out, 0, k);
  return 0;
}

void GroupedArrayFloat32_Head(GroupedArrayHandle handle, int k, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(Head<float>, k, out, 0, k);
}
void GroupedArrayFloat64_Head(GroupedArrayHandle handle, int k, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(Head<double>, k, out, 0, k);
}

void GroupedArrayFloat32_Tail(GroupedArrayHandle handle, int k, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(Tail<float>, k, out, 0, k);
}
void GroupedArrayFloat64_Tail(GroupedArrayHandle handle, int k, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(Tail<double>, k, out, 0, k);
}

void GroupedArrayFloat32_Append(GroupedArrayHandle handle,
                                GroupedArrayHandle other_handle,
                                const indptr_t *out_indptr, float *out_data) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  auto other = reinterpret_cast<const GroupedArray<float> *>(other_handle);
  ga->Zip(Append<float>, other, out_indptr, out_data);
}
void GroupedArrayFloat64_Append(GroupedArrayHandle handle,
                                GroupedArrayHandle other_handle,
                                const indptr_t *out_indptr, double *out_data) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  auto other = reinterpret_cast<const GroupedArray<double> *>(other_handle);
  ga->Zip(Append<double>, other, out_indptr, out_data);
}

void GroupedArrayFloat32_Tails(GroupedArrayHandle handle,
                               const indptr_t *indptr_out, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->VariableReduce(Tail<float>, indptr_out, out);
}
void GroupedArrayFloat64_Tails(GroupedArrayHandle handle,
                               const indptr_t *indptr_out, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->VariableReduce(Tail<double>, indptr_out, out);
}
