#include "c_api.h"

// Float32 Methods
// Single array
// Rolling
int Float32_RollingMeanTransform(float *data, int length, int window_size,
                                 int min_samples, float *out) {
  RollingMeanTransform<float>(data, length, out, window_size, min_samples);
  return 0;
}
int Float32_RollingStdTransform(float *data, int length, int window_size,
                                int min_samples, float *out) {
  RollingStdTransform<float>(data, length, out, window_size, min_samples);
  return 0;
}
int Float32_RollingMinTransform(float *data, int length, int window_size,
                                int min_samples, float *out) {
  RollingMinTransform<float>(data, length, out, window_size, min_samples);
  return 0;
}
int Float32_RollingMaxTransform(float *data, int length, int window_size,
                                int min_samples, float *out) {
  RollingMaxTransform<float>(data, length, out, window_size, min_samples);
  return 0;
}
int Float32_RollingQuantileTransform(float *data, int length, float p,
                                     int window_size, int min_samples,
                                     float *out) {
  RollingQuantileTransform<float>(data, length, out, window_size, min_samples,
                                  p);
  return 0;
}

// Seasonal rolling
int Float32_SeasonalRollingMeanTransform(float *data, int length,
                                         int season_length, int window_size,
                                         int min_samples, float *out) {
  SeasonalRollingMeanTransform<float>()(data, length, out, season_length,
                                        window_size, min_samples);
  return 0;
}
int Float32_SeasonalRollingStdTransform(float *data, int length,
                                        int season_length, int window_size,
                                        int min_samples, float *out) {
  SeasonalRollingStdTransform<float>()(data, length, out, season_length,
                                       window_size, min_samples);
  return 0;
}
int Float32_SeasonalRollingMinTransform(float *data, int length,
                                        int season_length, int window_size,
                                        int min_samples, float *out) {
  SeasonalRollingMinTransform<float>()(data, length, out, season_length,
                                       window_size, min_samples);
  return 0;
}
int Float32_SeasonalRollingMaxTransform(float *data, int length,
                                        int season_length, int window_size,
                                        int min_samples, float *out) {
  SeasonalRollingMaxTransform<float>()(data, length, out, season_length,
                                       window_size, min_samples);
  return 0;
}
int Float32_SeasonalRollingQuantileTransform(float *data, int length,
                                             int season_length, float p,
                                             int window_size, int min_samples,
                                             float *out) {
  SeasonalRollingQuantileTransform<float>()(data, length, out, season_length,
                                            window_size, min_samples, p);
  return 0;
}

// Expanding
int Float32_ExpandingMeanTransform(float *data, int length, float *out) {
  float tmp;
  ExpandingMeanTransform<float>(data, length, out, &tmp);
  return 0;
}
int Float32_ExpandingStdTransform(float *data, int length, float *out) {
  float tmp;
  RollingStdTransformWithStats(data, length, out, &tmp, false, length, 2);
  return 0;
}
int Float32_ExpandingMinTransform(float *data, int length, float *out) {
  ExpandingMinTransform<float>()(data, length, out);
  return 0;
}
int Float32_ExpandingMaxTransform(float *data, int length, float *out) {
  ExpandingMaxTransform<float>()(data, length, out);
  return 0;
}
int Float32_ExpandingQuantileTransform(float *data, int length, float p,
                                       float *out) {
  ExpandingQuantileTransform<float>(data, length, out, p);
  return 0;
}

// Exponentially weighted
int Float32_ExponentiallyWeightedMeanTransform(float *data, int length,
                                               float alpha, float *out) {
  ExponentiallyWeightedMeanTransform<float>(data, length, out, alpha);
  return 0;
}

// Scalers
float Float32_BoxCoxLambdaGuerrero(const float *x, int n, int period,
                                   float lower, float upper) {
  float out;
  BoxCoxLambda_Guerrero<float>(x, n, &out, period, lower, upper);
  return out;
}
float Float32_BoxCoxLambdaLogLik(const float *x, int n, float lower,
                                 float upper) {
  float out;
  BoxCoxLambda_LogLik<float>(x, n, &out, lower, upper);
  return out;
}
void Float32_BoxCoxTransform(const float *x, int n, float lambda, float *out) {
  std::transform(x, x + n, out, [lambda](float x) {
    return BoxCoxTransform<float>(x, lambda, 0.0);
  });
}
void Float32_BoxCoxInverseTransform(const float *x, int n, float lambda,
                                    float *out) {
  std::transform(x, x + n, out, [lambda](float x) {
    return BoxCoxInverseTransform<float>(x, lambda, 0.0);
  });
}

// Differences
void Float32_Difference(const float *x, indptr_t n, int d, float *out) {
  Difference<float>(x, n, out, d);
}
int Float32_NumDiffs(const float *x, indptr_t n, int max_d) {
  float out;
  NumDiffs(x, n, &out, max_d);
  return static_cast<int>(out);
}
int Float32_NumSeasDiffs(const float *x, indptr_t n, int period, int max_d) {
  float out;
  NumSeasDiffs(x, n, &out, period, max_d);
  return static_cast<int>(out);
}
int Float32_Period(const float *x, size_t n, int max_lag) {
  float tmp;
  GreatestAutocovariance(x, n, &tmp, max_lag);
  return static_cast<int>(tmp);
}

// GA
// Lag
int GroupedArrayFloat32_LagTransform(GroupedArrayHandle handle, int lag,
                                     float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(LagTransform<float>, lag, out);
  return 0;
}

// Manipulation
int GroupedArrayFloat32_Create(const float *data, indptr_t n_data,
                               indptr_t *indptr, indptr_t n_indptr,
                               int num_threads, GroupedArrayHandle *out) {
  *out = new GroupedArray<float>(data, n_data, indptr, n_indptr, num_threads);
  return 0;
}
int GroupedArrayFloat32_Delete(GroupedArrayHandle handle) {
  delete reinterpret_cast<GroupedArray<float> *>(handle);
  return 0;
}
int GroupedArrayFloat32_IndexFromEnd(GroupedArrayHandle handle, int k,
                                     float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(IndexFromEnd<float>, 1, out, 0, k);
  return 0;
}

void GroupedArrayFloat32_Head(GroupedArrayHandle handle, int k, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(Head<float>, k, out, 0, k);
}
void GroupedArrayFloat32_Tail(GroupedArrayHandle handle, int k, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(Tail<float>, k, out, 0, k);
}
void GroupedArrayFloat32_Append(GroupedArrayHandle handle,
                                GroupedArrayHandle other_handle,
                                const indptr_t *out_indptr, float *out_data) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  auto other = reinterpret_cast<const GroupedArray<float> *>(other_handle);
  ga->Zip(Append<float>, other, out_indptr, out_data);
}
void GroupedArrayFloat32_Tails(GroupedArrayHandle handle,
                               const indptr_t *indptr_out, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->VariableReduce(Tail<float>, indptr_out, out);
}

// Lag
// Rolling
int GroupedArrayFloat32_RollingMeanTransform(GroupedArrayHandle handle, int lag,
                                             int window_size, int min_samples,
                                             float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(RollingMeanTransform<float>, lag, out, window_size,
                min_samples);
  return 0;
}
int GroupedArrayFloat32_RollingStdTransform(GroupedArrayHandle handle, int lag,
                                            int window_size, int min_samples,
                                            float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(RollingStdTransform<float>, lag, out, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat32_RollingMinTransform(GroupedArrayHandle handle, int lag,
                                            int window_size, int min_samples,
                                            float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(RollingMinTransform<float>, lag, out, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat32_RollingMaxTransform(GroupedArrayHandle handle, int lag,
                                            int window_size, int min_samples,
                                            float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(RollingMaxTransform<float>, lag, out, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat32_RollingQuantileTransform(GroupedArrayHandle handle,
                                                 int lag, float p,
                                                 int window_size,
                                                 int min_samples, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(RollingQuantileTransform<float>, lag, out, window_size,
                min_samples, p);
  return 0;
}
int GroupedArrayFloat32_RollingMeanUpdate(GroupedArrayHandle handle, int lag,
                                          int window_size, int min_samples,
                                          float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RollingMeanUpdate<float>(), 1, out, lag, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat32_RollingStdUpdate(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RollingStdUpdate<float>(), 1, out, lag, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat32_RollingMinUpdate(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RollingMinUpdate<float>(), 1, out, lag, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat32_RollingMaxUpdate(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RollingMaxUpdate<float>(), 1, out, lag, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat32_RollingQuantileUpdate(GroupedArrayHandle handle,
                                              int lag, float p, int window_size,
                                              int min_samples, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RollingQuantileUpdate<float>(), 1, out, lag, window_size,
             min_samples, p);
  return 0;
}

// Seasonal rolling
int GroupedArrayFloat32_SeasonalRollingMeanTransform(GroupedArrayHandle handle,
                                                     int lag, int season_length,
                                                     int window_size,
                                                     int min_samples,
                                                     float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(SeasonalRollingMeanTransform<float>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}
int GroupedArrayFloat32_SeasonalRollingStdTransform(GroupedArrayHandle handle,
                                                    int lag, int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(SeasonalRollingStdTransform<float>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}
int GroupedArrayFloat32_SeasonalRollingMinTransform(GroupedArrayHandle handle,
                                                    int lag, int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(SeasonalRollingMinTransform<float>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}
int GroupedArrayFloat32_SeasonalRollingMaxTransform(GroupedArrayHandle handle,
                                                    int lag, int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(SeasonalRollingMaxTransform<float>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}
int GroupedArrayFloat32_SeasonalRollingQuantileTransform(
    GroupedArrayHandle handle, int lag, int season_length, float p,
    int window_size, int min_samples, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(SeasonalRollingQuantileTransform<float>(), lag, out,
                season_length, window_size, min_samples, p);
  return 0;
}
int GroupedArrayFloat32_SeasonalRollingMeanUpdate(GroupedArrayHandle handle,
                                                  int lag, int season_length,
                                                  int window_size,
                                                  int min_samples, float *out) {

  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(SeasonalRollingMeanUpdate<float>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}
int GroupedArrayFloat32_SeasonalRollingStdUpdate(GroupedArrayHandle handle,
                                                 int lag, int season_length,
                                                 int window_size,
                                                 int min_samples, float *out) {

  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(SeasonalRollingStdUpdate<float>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}
int GroupedArrayFloat32_SeasonalRollingMinUpdate(GroupedArrayHandle handle,
                                                 int lag, int season_length,
                                                 int window_size,
                                                 int min_samples, float *out) {

  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(SeasonalRollingMinUpdate<float>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}
int GroupedArrayFloat32_SeasonalRollingMaxUpdate(GroupedArrayHandle handle,
                                                 int lag, int season_length,
                                                 int window_size,
                                                 int min_samples, float *out) {

  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(SeasonalRollingMaxUpdate<float>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}
int GroupedArrayFloat32_SeasonalRollingQuantileUpdate(
    GroupedArrayHandle handle, int lag, int season_length, float p,
    int window_size, int min_samples, float *out) {

  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(SeasonalRollingQuantileUpdate<float>(), 1, out, lag, season_length,
             window_size, min_samples, p);
  return 0;
}

// Expanding
int GroupedArrayFloat32_ExpandingMeanTransform(GroupedArrayHandle handle,
                                               int lag, float *out,
                                               float *agg) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->TransformAndReduce(ExpandingMeanTransform<float>, lag, out, 1, agg);
  return 0;
}
int GroupedArrayFloat32_ExpandingStdTransform(GroupedArrayHandle handle,
                                              int lag, float *out, float *agg) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->TransformAndReduce(ExpandingStdTransform<float>, lag, out, 3, agg);
  return 0;
}
int GroupedArrayFloat32_ExpandingMinTransform(GroupedArrayHandle handle,
                                              int lag, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(ExpandingMinTransform<float>(), lag, out);
  return 0;
}
int GroupedArrayFloat32_ExpandingMaxTransform(GroupedArrayHandle handle,
                                              int lag, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(ExpandingMaxTransform<float>(), lag, out);

  return 0;
}
int GroupedArrayFloat32_ExpandingQuantileTransform(GroupedArrayHandle handle,
                                                   int lag, float p,
                                                   float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(ExpandingQuantileTransform<float>, lag, out, p);
  return 0;
}
int GroupedArrayFloat32_ExpandingQuantileUpdate(GroupedArrayHandle handle,
                                                int lag, float p, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(ExpandingQuantileUpdate<float>, 1, out, lag, p);
  return 0;
}

// Exponentially weighted
int GroupedArrayFloat32_ExponentiallyWeightedMeanTransform(
    GroupedArrayHandle handle, int lag, float alpha, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(ExponentiallyWeightedMeanTransform<float>, lag, out, alpha);
  return 0;
}

// Scalers
int GroupedArrayFloat32_MinMaxScalerStats(GroupedArrayHandle handle,
                                          float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(MinMaxScalerStats<float>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat32_StandardScalerStats(GroupedArrayHandle handle,
                                            float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(StandardScalerStats<float>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat32_RobustIqrScalerStats(GroupedArrayHandle handle,
                                             float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RobustScalerIqrStats<float>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat32_RobustMadScalerStats(GroupedArrayHandle handle,
                                             float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RobustScalerMadStats<float>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat32_ScalerTransform(GroupedArrayHandle handle,
                                        const float *stats, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->ScalerTransform(CommonScalerTransform<float>, stats, out);
  return 0;
}
int GroupedArrayFloat32_ScalerInverseTransform(GroupedArrayHandle handle,
                                               const float *stats, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->ScalerTransform(CommonScalerInverseTransform<float>, stats, out);
  return 0;
}
int GroupedArrayFloat32_BoxCoxLambdaGuerrero(GroupedArrayHandle handle,
                                             int period, float lower,
                                             float upper, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(BoxCoxLambda_Guerrero<float>, 2, out, 0, period, lower, upper);
  return 0;
}
void GroupedArrayFloat32_BoxCoxLambdaLogLik(GroupedArrayHandle handle,
                                            float lower, float upper,
                                            float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(BoxCoxLambda_LogLik<float>, 2, out, 0, lower, upper);
}
int GroupedArrayFloat32_BoxCoxTransform(GroupedArrayHandle handle,
                                        const float *lambdas, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->ScalerTransform(BoxCoxTransform<float>, lambdas, out);
  return 0;
}
int GroupedArrayFloat32_BoxCoxInverseTransform(GroupedArrayHandle handle,
                                               const float *lambdas,
                                               float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->ScalerTransform(BoxCoxInverseTransform<float>, lambdas, out);
  return 0;
}

// Differences
void GroupedArrayFloat32_NumDiffs(GroupedArrayHandle handle, int max_d,
                                  float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(NumDiffs<float>, 1, out, 0, max_d);
}
void GroupedArrayFloat32_NumSeasDiffs(GroupedArrayHandle handle, int period,
                                      int max_d, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(NumSeasDiffs<float>, 1, out, 0, period, max_d);
}
void GroupedArrayFloat32_NumSeasDiffsPeriods(GroupedArrayHandle handle,
                                             int max_d,
                                             float *periods_and_out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(NumSeasDiffsPeriods<float>, 2, periods_and_out, 0, max_d);
}

void GroupedArrayFloat32_Period(GroupedArrayHandle handle, size_t max_lag,
                                float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(GreatestAutocovariance<float>, 1, out, 0, max_lag);
}
void GroupedArrayFloat32_Difference(GroupedArrayHandle handle, int d,
                                    float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(Difference<float>, 0, out, d);
}
void GroupedArrayFloat32_Differences(GroupedArrayHandle handle,
                                     const indptr_t *ds, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->VariableTransform(Differences<float>, ds, out);
}
void GroupedArrayFloat32_InvertDifferences(GroupedArrayHandle handle,
                                           GroupedArrayHandle tails_handle,
                                           const indptr_t *out_indptr,
                                           float *out_data) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  auto tails_ga = reinterpret_cast<const GroupedArray<float> *>(tails_handle);
  ga->Zip(InvertDifference<float>, tails_ga, out_indptr, out_data);
}

// Float64 Methods
//  Single array
//  Rolling
int Float64_RollingMeanTransform(double *data, int length, int window_size,
                                 int min_samples, double *out) {
  RollingMeanTransform<double>(data, length, out, window_size, min_samples);
  return 0;
}
int Float64_RollingStdTransform(double *data, int length, int window_size,
                                int min_samples, double *out) {
  RollingStdTransform<double>(data, length, out, window_size, min_samples);
  return 0;
}
int Float64_RollingMinTransform(double *data, int length, int window_size,
                                int min_samples, double *out) {
  RollingMinTransform<double>(data, length, out, window_size, min_samples);
  return 0;
}
int Float64_RollingMaxTransform(double *data, int length, int window_size,
                                int min_samples, double *out) {
  RollingMaxTransform<double>(data, length, out, window_size, min_samples);
  return 0;
}
int Float64_RollingQuantileTransform(double *data, int length, double p,
                                     int window_size, int min_samples,
                                     double *out) {
  RollingQuantileTransform<double>(data, length, out, window_size, min_samples,
                                   p);
  return 0;
}

// Seasonal rolling
int Float64_SeasonalRollingMeanTransform(double *data, int length,
                                         int season_length, int window_size,
                                         int min_samples, double *out) {
  SeasonalRollingMeanTransform<double>()(data, length, out, season_length,
                                         window_size, min_samples);
  return 0;
}
int Float64_SeasonalRollingStdTransform(double *data, int length,
                                        int season_length, int window_size,
                                        int min_samples, double *out) {
  SeasonalRollingStdTransform<double>()(data, length, out, season_length,
                                        window_size, min_samples);
  return 0;
}
int Float64_SeasonalRollingMinTransform(double *data, int length,
                                        int season_length, int window_size,
                                        int min_samples, double *out) {
  SeasonalRollingMinTransform<double>()(data, length, out, season_length,
                                        window_size, min_samples);
  return 0;
}
int Float64_SeasonalRollingMaxTransform(double *data, int length,
                                        int season_length, int window_size,
                                        int min_samples, double *out) {
  SeasonalRollingMaxTransform<double>()(data, length, out, season_length,
                                        window_size, min_samples);
  return 0;
}
int Float64_SeasonalRollingQuantileTransform(double *data, int length,
                                             int season_length, double p,
                                             int window_size, int min_samples,
                                             double *out) {
  SeasonalRollingQuantileTransform<double>()(data, length, out, season_length,
                                             window_size, min_samples, p);
  return 0;
}

// Expanding
int Float64_ExpandingMeanTransform(double *data, int length, double *out) {
  double tmp;
  ExpandingMeanTransform<double>(data, length, out, &tmp);
  return 0;
}
int Float64_ExpandingStdTransform(double *data, int length, double *out) {
  double tmp;
  RollingStdTransformWithStats(data, length, out, &tmp, false, length, 2);
  return 0;
}
int Float64_ExpandingMinTransform(double *data, int length, double *out) {
  ExpandingMinTransform<double>()(data, length, out);
  return 0;
}
int Float64_ExpandingMaxTransform(double *data, int length, double *out) {
  ExpandingMaxTransform<double>()(data, length, out);
  return 0;
}
int Float64_ExpandingQuantileTransform(double *data, int length, double p,
                                       double *out) {
  ExpandingQuantileTransform<double>(data, length, out, p);
  return 0;
}

// Exponentially weighted
int Float64_ExponentiallyWeightedMeanTransform(double *data, int length,
                                               double alpha, double *out) {
  ExponentiallyWeightedMeanTransform<double>(data, length, out, alpha);
  return 0;
}

// Scalers
double Float64_BoxCoxLambdaGuerrero(const double *x, int n, int period,
                                    double lower, double upper) {
  double out;
  BoxCoxLambda_Guerrero<double>(x, n, &out, period, lower, upper);
  return out;
}
double Float64_BoxCoxLambdaLogLik(const double *x, int n, double lower,
                                  double upper) {
  double out;
  BoxCoxLambda_LogLik<double>(x, n, &out, lower, upper);
  return out;
}
void Float64_BoxCoxTransform(const double *x, int n, double lambda,
                             double *out) {
  std::transform(x, x + n, out, [lambda](double x) {
    return BoxCoxTransform<double>(x, lambda, 0.0);
  });
}
void Float64_BoxCoxInverseTransform(const double *x, int n, double lambda,
                                    double *out) {
  std::transform(x, x + n, out, [lambda](double x) {
    return BoxCoxInverseTransform<double>(x, lambda, 0.0);
  });
}

// Differences
void Float64_Difference(const double *x, indptr_t n, int d, double *out) {
  Difference<double>(x, n, out, d);
}
int Float64_NumDiffs(const double *x, indptr_t n, int max_d) {
  double out;
  NumDiffs(x, n, &out, max_d);
  return static_cast<int>(out);
}
int Float64_NumSeasDiffs(const double *x, indptr_t n, int period, int max_d) {
  double out;
  NumSeasDiffs(x, n, &out, period, max_d);
  return static_cast<int>(out);
}
int Float64_Period(const double *x, size_t n, int max_lag) {
  double tmp;
  GreatestAutocovariance(x, n, &tmp, max_lag);
  return static_cast<int>(tmp);
}

// GA
// Lag
int GroupedArrayFloat64_LagTransform(GroupedArrayHandle handle, int lag,
                                     double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(LagTransform<double>, lag, out);
  return 0;
}

// Manipulation
int GroupedArrayFloat64_Create(const double *data, indptr_t n_data,
                               indptr_t *indptr, indptr_t n_indptr,
                               int num_threads, GroupedArrayHandle *out) {
  *out = new GroupedArray<double>(data, n_data, indptr, n_indptr, num_threads);
  return 0;
}
int GroupedArrayFloat64_Delete(GroupedArrayHandle handle) {
  delete reinterpret_cast<GroupedArray<double> *>(handle);
  return 0;
}
int GroupedArrayFloat64_IndexFromEnd(GroupedArrayHandle handle, int k,
                                     double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(IndexFromEnd<double>, 1, out, 0, k);
  return 0;
}

void GroupedArrayFloat64_Head(GroupedArrayHandle handle, int k, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(Head<double>, k, out, 0, k);
}
void GroupedArrayFloat64_Tail(GroupedArrayHandle handle, int k, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(Tail<double>, k, out, 0, k);
}
void GroupedArrayFloat64_Append(GroupedArrayHandle handle,
                                GroupedArrayHandle other_handle,
                                const indptr_t *out_indptr, double *out_data) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  auto other = reinterpret_cast<const GroupedArray<double> *>(other_handle);
  ga->Zip(Append<double>, other, out_indptr, out_data);
}
void GroupedArrayFloat64_Tails(GroupedArrayHandle handle,
                               const indptr_t *indptr_out, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->VariableReduce(Tail<double>, indptr_out, out);
}

// Lag
// Rolling
int GroupedArrayFloat64_RollingMeanTransform(GroupedArrayHandle handle, int lag,
                                             int window_size, int min_samples,
                                             double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(RollingMeanTransform<double>, lag, out, window_size,
                min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingStdTransform(GroupedArrayHandle handle, int lag,
                                            int window_size, int min_samples,
                                            double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(RollingStdTransform<double>, lag, out, window_size,
                min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingMinTransform(GroupedArrayHandle handle, int lag,
                                            int window_size, int min_samples,
                                            double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(RollingMinTransform<double>, lag, out, window_size,
                min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingMaxTransform(GroupedArrayHandle handle, int lag,
                                            int window_size, int min_samples,
                                            double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(RollingMaxTransform<double>, lag, out, window_size,
                min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingQuantileTransform(GroupedArrayHandle handle,
                                                 int lag, double p,
                                                 int window_size,
                                                 int min_samples, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(RollingQuantileTransform<double>, lag, out, window_size,
                min_samples, p);
  return 0;
}
int GroupedArrayFloat64_RollingMeanUpdate(GroupedArrayHandle handle, int lag,
                                          int window_size, int min_samples,
                                          double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RollingMeanUpdate<double>(), 1, out, lag, window_size,
             min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingStdUpdate(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RollingStdUpdate<double>(), 1, out, lag, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingMinUpdate(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RollingMinUpdate<double>(), 1, out, lag, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingMaxUpdate(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RollingMaxUpdate<double>(), 1, out, lag, window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_RollingQuantileUpdate(GroupedArrayHandle handle,
                                              int lag, double p,
                                              int window_size, int min_samples,
                                              double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RollingQuantileUpdate<double>(), 1, out, lag, window_size,
             min_samples, p);
  return 0;
}

// Seasonal rolling
int GroupedArrayFloat64_SeasonalRollingMeanTransform(GroupedArrayHandle handle,
                                                     int lag, int season_length,
                                                     int window_size,
                                                     int min_samples,
                                                     double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(SeasonalRollingMeanTransform<double>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingStdTransform(GroupedArrayHandle handle,
                                                    int lag, int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(SeasonalRollingStdTransform<double>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingMinTransform(GroupedArrayHandle handle,
                                                    int lag, int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(SeasonalRollingMinTransform<double>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingMaxTransform(GroupedArrayHandle handle,
                                                    int lag, int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(SeasonalRollingMaxTransform<double>(), lag, out, season_length,
                window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingQuantileTransform(
    GroupedArrayHandle handle, int lag, int season_length, double p,
    int window_size, int min_samples, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(SeasonalRollingQuantileTransform<double>(), lag, out,
                season_length, window_size, min_samples, p);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingMeanUpdate(GroupedArrayHandle handle,
                                                  int lag, int season_length,
                                                  int window_size,
                                                  int min_samples,
                                                  double *out) {

  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(SeasonalRollingMeanUpdate<double>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingStdUpdate(GroupedArrayHandle handle,
                                                 int lag, int season_length,
                                                 int window_size,
                                                 int min_samples, double *out) {

  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(SeasonalRollingStdUpdate<double>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingMinUpdate(GroupedArrayHandle handle,
                                                 int lag, int season_length,
                                                 int window_size,
                                                 int min_samples, double *out) {

  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(SeasonalRollingMinUpdate<double>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingMaxUpdate(GroupedArrayHandle handle,
                                                 int lag, int season_length,
                                                 int window_size,
                                                 int min_samples, double *out) {

  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(SeasonalRollingMaxUpdate<double>(), 1, out, lag, season_length,
             window_size, min_samples);
  return 0;
}
int GroupedArrayFloat64_SeasonalRollingQuantileUpdate(
    GroupedArrayHandle handle, int lag, int season_length, double p,
    int window_size, int min_samples, double *out) {

  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(SeasonalRollingQuantileUpdate<double>(), 1, out, lag,
             season_length, window_size, min_samples, p);
  return 0;
}

// Expanding
int GroupedArrayFloat64_ExpandingMeanTransform(GroupedArrayHandle handle,
                                               int lag, double *out,
                                               double *agg) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->TransformAndReduce(ExpandingMeanTransform<double>, lag, out, 1, agg);
  return 0;
}
int GroupedArrayFloat64_ExpandingStdTransform(GroupedArrayHandle handle,
                                              int lag, double *out,
                                              double *agg) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->TransformAndReduce(ExpandingStdTransform<double>, lag, out, 3, agg);
  return 0;
}
int GroupedArrayFloat64_ExpandingMinTransform(GroupedArrayHandle handle,
                                              int lag, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(ExpandingMinTransform<double>(), lag, out);
  return 0;
}
int GroupedArrayFloat64_ExpandingMaxTransform(GroupedArrayHandle handle,
                                              int lag, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(ExpandingMaxTransform<double>(), lag, out);

  return 0;
}
int GroupedArrayFloat64_ExpandingQuantileTransform(GroupedArrayHandle handle,
                                                   int lag, double p,
                                                   double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(ExpandingQuantileTransform<double>, lag, out, p);
  return 0;
}
int GroupedArrayFloat64_ExpandingQuantileUpdate(GroupedArrayHandle handle,
                                                int lag, double p,
                                                double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(ExpandingQuantileUpdate<double>, 1, out, lag, p);
  return 0;
}

// Exponentially weighted
int GroupedArrayFloat64_ExponentiallyWeightedMeanTransform(
    GroupedArrayHandle handle, int lag, double alpha, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(ExponentiallyWeightedMeanTransform<double>, lag, out, alpha);
  return 0;
}

// Scalers
int GroupedArrayFloat64_MinMaxScalerStats(GroupedArrayHandle handle,
                                          double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(MinMaxScalerStats<double>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat64_StandardScalerStats(GroupedArrayHandle handle,
                                            double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(StandardScalerStats<double>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat64_RobustIqrScalerStats(GroupedArrayHandle handle,
                                             double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RobustScalerIqrStats<double>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat64_RobustMadScalerStats(GroupedArrayHandle handle,
                                             double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RobustScalerMadStats<double>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat64_ScalerTransform(GroupedArrayHandle handle,
                                        const double *stats, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->ScalerTransform(CommonScalerTransform<double>, stats, out);
  return 0;
}
int GroupedArrayFloat64_ScalerInverseTransform(GroupedArrayHandle handle,
                                               const double *stats,
                                               double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->ScalerTransform(CommonScalerInverseTransform<double>, stats, out);
  return 0;
}
int GroupedArrayFloat64_BoxCoxLambdaGuerrero(GroupedArrayHandle handle,
                                             int period, double lower,
                                             double upper, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(BoxCoxLambda_Guerrero<double>, 2, out, 0, period, lower, upper);
  return 0;
}
void GroupedArrayFloat64_BoxCoxLambdaLogLik(GroupedArrayHandle handle,
                                            double lower, double upper,
                                            double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(BoxCoxLambda_LogLik<double>, 2, out, 0, lower, upper);
}
int GroupedArrayFloat64_BoxCoxTransform(GroupedArrayHandle handle,
                                        const double *lambdas, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->ScalerTransform(BoxCoxTransform<double>, lambdas, out);
  return 0;
}
int GroupedArrayFloat64_BoxCoxInverseTransform(GroupedArrayHandle handle,
                                               const double *lambdas,
                                               double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->ScalerTransform(BoxCoxInverseTransform<double>, lambdas, out);
  return 0;
}

// Differences
void GroupedArrayFloat64_NumDiffs(GroupedArrayHandle handle, int max_d,
                                  double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(NumDiffs<double>, 1, out, 0, max_d);
}
void GroupedArrayFloat64_NumSeasDiffs(GroupedArrayHandle handle, int period,
                                      int max_d, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(NumSeasDiffs<double>, 1, out, 0, period, max_d);
}
void GroupedArrayFloat64_NumSeasDiffsPeriods(GroupedArrayHandle handle,
                                             int max_d,
                                             double *periods_and_out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(NumSeasDiffsPeriods<double>, 2, periods_and_out, 0, max_d);
}

void GroupedArrayFloat64_Period(GroupedArrayHandle handle, size_t max_lag,
                                double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(GreatestAutocovariance<double>, 1, out, 0, max_lag);
}
void GroupedArrayFloat64_Difference(GroupedArrayHandle handle, int d,
                                    double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(Difference<double>, 0, out, d);
}
void GroupedArrayFloat64_Differences(GroupedArrayHandle handle,
                                     const indptr_t *ds, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->VariableTransform(Differences<double>, ds, out);
}
void GroupedArrayFloat64_InvertDifferences(GroupedArrayHandle handle,
                                           GroupedArrayHandle tails_handle,
                                           const indptr_t *out_indptr,
                                           double *out_data) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  auto tails_ga = reinterpret_cast<const GroupedArray<double> *>(tails_handle);
  ga->Zip(InvertDifference<double>, tails_ga, out_indptr, out_data);
}
