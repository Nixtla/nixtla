#include "diff.h"
#include "kpss.h"
#include "seasonal.h"

template <typename T> inline bool IsConstant(const T *data, int n) {
  for (int i = 1; i < n; ++i) {
    if (data[i] != data[0]) {
      return false;
    }
  }
  return true;
}

template <typename T>
void InvertDifference(const T *data, int n, const T *tails, int d, T *out) {
  if (d == 0) {
    std::copy(data, data + n, out);
    return;
  }
  int upper = std::min(d, n);
  for (int i = 0; i < upper; ++i) {
    out[i] = data[i] + tails[i];
  }
  for (int i = upper; i < n; ++i) {
    out[i] = data[i] + out[i - d];
  }
}

template <typename T> void NumDiffs(const T *x, indptr_t n, T *out, int max_d) {
  // assume there are only NaNs at the start
  indptr_t start_idx = FirstNotNaN(x, n);
  x += start_idx;
  n -= start_idx;
  if (n < 3) {
    *out = 0;
    return;
  }
  constexpr T threshold = 0.463; // alpha = 0.05
  int d = 0;
  int n_lags = std::floor(3 * std::sqrt(n) / 13);
  bool do_diff = KPSS(x, n, n_lags) > threshold;
  std::vector<T> x_vec(n);
  std::copy(x, x + n, x_vec.begin());
  std::vector<T> diff_x(n);
  while (do_diff && d < max_d) {
    ++d;
    Difference(x_vec.data(), x_vec.size(), diff_x.data(), 1);
    if (IsConstant(diff_x.data() + d, diff_x.size() - d)) {
      *out = d;
      return;
    }
    std::copy(diff_x.begin(), diff_x.end(), x_vec.begin());
    if (n > d) {
      // we've taken d differences, so we have d NaNs
      do_diff = KPSS(x_vec.data() + d, n - d, n_lags) > threshold;
    } else {
      do_diff = false;
    }
  }
  *out = d;
  return;
}

template <typename T>
void NumSeasDiffs(const T *x, indptr_t n, T *out, int period, int max_d) {
  // assume there are only NaNs at the start
  indptr_t start_idx = FirstNotNaN(x, n);
  x += start_idx;
  n -= start_idx;
  if (n < 2 * period) {
    *out = 0;
    return;
  }
  constexpr T threshold = 0.64;
  int d = 0;
  bool do_diff = SeasHeuristic(x, n, period) > threshold;
  std::vector<T> x_vec(n);
  std::copy(x, x + n, x_vec.begin());
  std::vector<T> diff_x(n);
  while (do_diff && d < max_d) {
    ++d;
    Difference(x_vec.data(), x_vec.size(), diff_x.data(), period);
    if (IsConstant(diff_x.data() + d * period, n - d * period)) {
      *out = d;
      return;
    }
    std::copy(diff_x.begin(), diff_x.end(), x_vec.begin());
    // we'll have d * period NaNs and we need 2 * period samples for the STL
    if (n > (d + 2) * period && d < max_d) {
      do_diff = SeasHeuristic(x_vec.data() + d * period, n - d * period,
                              period) > threshold;
    } else {
      do_diff = false;
    }
  }
  *out = d;
  return;
}

template <typename T>
void NumSeasDiffsPeriods(const T *x, indptr_t n, T *period_and_out, int max_d) {
  int period = static_cast<int>(period_and_out[0]);
  T *out = period_and_out + 1;
  NumSeasDiffs(x, n, out, period, max_d);
}

void Float32_Difference(const float *x, indptr_t n, int d, float *out) {
  Difference<float>(x, n, out, d);
}
void Float64_Difference(const double *x, indptr_t n, int d, double *out) {
  Difference<double>(x, n, out, d);
}

int Float32_NumDiffs(const float *x, indptr_t n, int max_d) {
  float out;
  NumDiffs(x, n, &out, max_d);
  return static_cast<int>(out);
}
int Float64_NumDiffs(const double *x, indptr_t n, int max_d) {
  double out;
  NumDiffs(x, n, &out, max_d);
  return static_cast<int>(out);
}

int Float32_NumSeasDiffs(const float *x, indptr_t n, int period, int max_d) {
  float out;
  NumSeasDiffs(x, n, &out, period, max_d);
  return static_cast<int>(out);
}
int Float64_NumSeasDiffs(const double *x, indptr_t n, int period, int max_d) {
  double out;
  NumSeasDiffs(x, n, &out, period, max_d);
  return static_cast<int>(out);
}

int Float32_Period(const float *x, size_t n, int max_lag) {
  float tmp;
  GreatestAutocovariance(x, n, &tmp, max_lag);
  return static_cast<int>(tmp);
}
int Float64_Period(const double *x, size_t n, int max_lag) {
  double tmp;
  GreatestAutocovariance(x, n, &tmp, max_lag);
  return static_cast<int>(tmp);
}

void GroupedArrayFloat32_NumDiffs(GroupedArrayHandle handle, int max_d,
                                  float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(NumDiffs<float>, 1, out, 0, max_d);
}
void GroupedArrayFloat64_NumDiffs(GroupedArrayHandle handle, int max_d,
                                  double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(NumDiffs<double>, 1, out, 0, max_d);
}

void GroupedArrayFloat32_NumSeasDiffs(GroupedArrayHandle handle, int period,
                                      int max_d, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(NumSeasDiffs<float>, 1, out, 0, period, max_d);
}
void GroupedArrayFloat64_NumSeasDiffs(GroupedArrayHandle handle, int period,
                                      int max_d, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(NumSeasDiffs<double>, 1, out, 0, period, max_d);
}

void GroupedArrayFloat32_NumSeasDiffsPeriods(GroupedArrayHandle handle,
                                             int max_d,
                                             float *periods_and_out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(NumSeasDiffsPeriods<float>, 2, periods_and_out, 0, max_d);
}
void GroupedArrayFloat64_NumSeasDiffsPeriods(GroupedArrayHandle handle,
                                             int max_d,
                                             double *periods_and_out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(NumSeasDiffsPeriods<double>, 2, periods_and_out, 0, max_d);
}

void GroupedArrayFloat32_Period(GroupedArrayHandle handle, size_t max_lag,
                                float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(GreatestAutocovariance<float>, 1, out, 0, max_lag);
}
void GroupedArrayFloat64_Period(GroupedArrayHandle handle, size_t max_lag,
                                double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(GreatestAutocovariance<double>, 1, out, 0, max_lag);
}

void GroupedArrayFloat32_Difference(GroupedArrayHandle handle, int d,
                                    float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(Difference<float>, 0, out, d);
}
void GroupedArrayFloat64_Difference(GroupedArrayHandle handle, int d,
                                    double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(Difference<double>, 0, out, d);
}

void GroupedArrayFloat32_Differences(GroupedArrayHandle handle,
                                     const indptr_t *ds, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->VariableTransform(Differences<float>, ds, out);
}
void GroupedArrayFloat64_Differences(GroupedArrayHandle handle,
                                     const indptr_t *ds, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->VariableTransform(Differences<double>, ds, out);
}

void GroupedArrayFloat32_InvertDifferences(GroupedArrayHandle handle,
                                           GroupedArrayHandle tails_handle,
                                           const indptr_t *out_indptr,
                                           float *out_data) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  auto tails_ga = reinterpret_cast<const GroupedArray<float> *>(tails_handle);
  ga->Zip(InvertDifference<float>, tails_ga, out_indptr, out_data);
}
void GroupedArrayFloat64_InvertDifferences(GroupedArrayHandle handle,
                                           GroupedArrayHandle tails_handle,
                                           const indptr_t *out_indptr,
                                           double *out_data) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  auto tails_ga = reinterpret_cast<const GroupedArray<double> *>(tails_handle);
  ga->Zip(InvertDifference<double>, tails_ga, out_indptr, out_data);
}
