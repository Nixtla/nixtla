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
void GroupedArrayFloat32_LagTransform(const float *data, const indptr_t *indptr,
                                      int n_indptr, int num_threads, int lag,
                                      float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Transform(LagTransform<float>, lag, out);
}

// Manipulation
void GroupedArrayFloat32_IndexFromEnd(const float *data, const indptr_t *indptr,
                                      int n_indptr, int num_threads, int k,
                                      float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(IndexFromEnd<float>, 1, out, 0, k);
}

void GroupedArrayFloat32_Head(const float *data, const indptr_t *indptr,
                              int n_indptr, int num_threads, int k,
                              float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(Head<float>, k, out, 0, k);
}
void GroupedArrayFloat32_Tail(const float *data, const indptr_t *indptr,
                              int n_indptr, int num_threads, int k,
                              float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(Tail<float>, k, out, 0, k);
}
void GroupedArrayFloat32_Append(const float *data, const indptr_t *indptr,
                                int n_indptr, int num_threads,
                                const float *other_data,
                                const indptr_t *other_indptr,
                                int other_n_indptr, const indptr_t *out_indptr,
                                float *out_data) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  auto other = GroupedArray<float>(other_data, other_indptr, other_n_indptr,
                                   num_threads);
  ga.Zip(Append<float>, other, out_indptr, out_data);
}
void GroupedArrayFloat32_Tails(const float *data, const indptr_t *indptr,
                               int n_indptr, int num_threads,
                               const indptr_t *indptr_out, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.VariableReduce(Tail<float>, indptr_out, out);
}

// Lag
// Rolling
void GroupedArrayFloat32_RollingMeanTransform(const float *data,
                                              const indptr_t *indptr,
                                              int n_indptr, int num_threads,
                                              int lag, int window_size,
                                              int min_samples, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Transform(RollingMeanTransform<float>, lag, out, window_size, min_samples);
}
void GroupedArrayFloat32_RollingStdTransform(const float *data,
                                             const indptr_t *indptr,
                                             int n_indptr, int num_threads,
                                             int lag, int window_size,
                                             int min_samples, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Transform(RollingStdTransform<float>, lag, out, window_size, min_samples);
}
void GroupedArrayFloat32_RollingMinTransform(const float *data,
                                             const indptr_t *indptr,
                                             int n_indptr, int num_threads,
                                             int lag, int window_size,
                                             int min_samples, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Transform(RollingMinTransform<float>, lag, out, window_size, min_samples);
}
void GroupedArrayFloat32_RollingMaxTransform(const float *data,
                                             const indptr_t *indptr,
                                             int n_indptr, int num_threads,
                                             int lag, int window_size,
                                             int min_samples, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Transform(RollingMaxTransform<float>, lag, out, window_size, min_samples);
}
void GroupedArrayFloat32_RollingQuantileTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, float p, int window_size, int min_samples, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Transform(RollingQuantileTransform<float>, lag, out, window_size,
               min_samples, p);
}
void GroupedArrayFloat32_RollingMeanUpdate(const float *data,
                                           const indptr_t *indptr, int n_indptr,
                                           int num_threads, int lag,
                                           int window_size, int min_samples,
                                           float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(RollingMeanUpdate<float>(), 1, out, lag, window_size, min_samples);
}
void GroupedArrayFloat32_RollingStdUpdate(const float *data,
                                          const indptr_t *indptr, int n_indptr,
                                          int num_threads, int lag,
                                          int window_size, int min_samples,
                                          float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(RollingStdUpdate<float>(), 1, out, lag, window_size, min_samples);
}
void GroupedArrayFloat32_RollingMinUpdate(const float *data,
                                          const indptr_t *indptr, int n_indptr,
                                          int num_threads, int lag,
                                          int window_size, int min_samples,
                                          float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(RollingMinUpdate<float>(), 1, out, lag, window_size, min_samples);
}
void GroupedArrayFloat32_RollingMaxUpdate(const float *data,
                                          const indptr_t *indptr, int n_indptr,
                                          int num_threads, int lag,
                                          int window_size, int min_samples,
                                          float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(RollingMaxUpdate<float>(), 1, out, lag, window_size, min_samples);
}
void GroupedArrayFloat32_RollingQuantileUpdate(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, float p, int window_size, int min_samples, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(RollingQuantileUpdate<float>(), 1, out, lag, window_size,
            min_samples, p);
}

// Seasonal rolling
void GroupedArrayFloat32_SeasonalRollingMeanTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Transform(SeasonalRollingMeanTransform<float>(), lag, out, season_length,
               window_size, min_samples);
}
void GroupedArrayFloat32_SeasonalRollingStdTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Transform(SeasonalRollingStdTransform<float>(), lag, out, season_length,
               window_size, min_samples);
}
void GroupedArrayFloat32_SeasonalRollingMinTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Transform(SeasonalRollingMinTransform<float>(), lag, out, season_length,
               window_size, min_samples);
}
void GroupedArrayFloat32_SeasonalRollingMaxTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Transform(SeasonalRollingMaxTransform<float>(), lag, out, season_length,
               window_size, min_samples);
}
void GroupedArrayFloat32_SeasonalRollingQuantileTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, float p, int window_size, int min_samples,
    float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Transform(SeasonalRollingQuantileTransform<float>(), lag, out,
               season_length, window_size, min_samples, p);
}
void GroupedArrayFloat32_SeasonalRollingMeanUpdate(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(SeasonalRollingMeanUpdate<float>(), 1, out, lag, season_length,
            window_size, min_samples);
}
void GroupedArrayFloat32_SeasonalRollingStdUpdate(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(SeasonalRollingStdUpdate<float>(), 1, out, lag, season_length,
            window_size, min_samples);
}
void GroupedArrayFloat32_SeasonalRollingMinUpdate(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(SeasonalRollingMinUpdate<float>(), 1, out, lag, season_length,
            window_size, min_samples);
}
void GroupedArrayFloat32_SeasonalRollingMaxUpdate(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(SeasonalRollingMaxUpdate<float>(), 1, out, lag, season_length,
            window_size, min_samples);
}
void GroupedArrayFloat32_SeasonalRollingQuantileUpdate(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, float p, int window_size, int min_samples,
    float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(SeasonalRollingQuantileUpdate<float>(), 1, out, lag, season_length,
            window_size, min_samples, p);
}

// Expanding
void GroupedArrayFloat32_ExpandingMeanTransform(const float *data,
                                                const indptr_t *indptr,
                                                int n_indptr, int num_threads,
                                                int lag, float *out,
                                                float *agg) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.TransformAndReduce(ExpandingMeanTransform<float>, lag, out, 1, agg);
}
void GroupedArrayFloat32_ExpandingStdTransform(const float *data,
                                               const indptr_t *indptr,
                                               int n_indptr, int num_threads,
                                               int lag, float *out,
                                               float *agg) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.TransformAndReduce(ExpandingStdTransform<float>, lag, out, 3, agg);
}
void GroupedArrayFloat32_ExpandingMinTransform(const float *data,
                                               const indptr_t *indptr,
                                               int n_indptr, int num_threads,
                                               int lag, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Transform(ExpandingMinTransform<float>(), lag, out);
}
void GroupedArrayFloat32_ExpandingMaxTransform(const float *data,
                                               const indptr_t *indptr,
                                               int n_indptr, int num_threads,
                                               int lag, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Transform(ExpandingMaxTransform<float>(), lag, out);
}
void GroupedArrayFloat32_ExpandingQuantileTransform(const float *data,
                                                    const indptr_t *indptr,
                                                    int n_indptr,
                                                    int num_threads, int lag,
                                                    float p, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Transform(ExpandingQuantileTransform<float>, lag, out, p);
}
void GroupedArrayFloat32_ExpandingQuantileUpdate(const float *data,
                                                 const indptr_t *indptr,
                                                 int n_indptr, int num_threads,
                                                 int lag, float p, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(ExpandingQuantileUpdate<float>, 1, out, lag, p);
}

// Exponentially weighted
void GroupedArrayFloat32_ExponentiallyWeightedMeanTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, float alpha, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Transform(ExponentiallyWeightedMeanTransform<float>, lag, out, alpha);
}

// Scalers
void GroupedArrayFloat32_MinMaxScalerStats(const float *data,
                                           const indptr_t *indptr, int n_indptr,
                                           int num_threads, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(MinMaxScalerStats<float>, 2, out, 0);
}
void GroupedArrayFloat32_StandardScalerStats(const float *data,
                                             const indptr_t *indptr,
                                             int n_indptr, int num_threads,
                                             float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(StandardScalerStats<float>, 2, out, 0);
}
void GroupedArrayFloat32_RobustIqrScalerStats(const float *data,
                                              const indptr_t *indptr,
                                              int n_indptr, int num_threads,
                                              float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(RobustScalerIqrStats<float>, 2, out, 0);
}
void GroupedArrayFloat32_RobustMadScalerStats(const float *data,
                                              const indptr_t *indptr,
                                              int n_indptr, int num_threads,
                                              float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(RobustScalerMadStats<float>, 2, out, 0);
}
void GroupedArrayFloat32_ScalerTransform(const float *data,
                                         const indptr_t *indptr, int n_indptr,
                                         int num_threads, const float *stats,
                                         float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.ScalerTransform(CommonScalerTransform<float>, stats, out);
}
void GroupedArrayFloat32_ScalerInverseTransform(const float *data,
                                                const indptr_t *indptr,
                                                int n_indptr, int num_threads,
                                                const float *stats,
                                                float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.ScalerTransform(CommonScalerInverseTransform<float>, stats, out);
}
void GroupedArrayFloat32_BoxCoxLambdaGuerrero(const float *data,
                                              const indptr_t *indptr,
                                              int n_indptr, int num_threads,
                                              int period, float lower,
                                              float upper, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(BoxCoxLambda_Guerrero<float>, 2, out, 0, period, lower, upper);
}
void GroupedArrayFloat32_BoxCoxLambdaLogLik(const float *data,
                                            const indptr_t *indptr,
                                            int n_indptr, int num_threads,
                                            float lower, float upper,
                                            float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(BoxCoxLambda_LogLik<float>, 2, out, 0, lower, upper);
}
void GroupedArrayFloat32_BoxCoxTransform(const float *data,
                                         const indptr_t *indptr, int n_indptr,
                                         int num_threads, const float *lambdas,
                                         float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.ScalerTransform(BoxCoxTransform<float>, lambdas, out);
}
void GroupedArrayFloat32_BoxCoxInverseTransform(const float *data,
                                                const indptr_t *indptr,
                                                int n_indptr, int num_threads,
                                                const float *lambdas,
                                                float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.ScalerTransform(BoxCoxInverseTransform<float>, lambdas, out);
}

// Differences
void GroupedArrayFloat32_NumDiffs(const float *data, const indptr_t *indptr,
                                  int n_indptr, int num_threads, int max_d,
                                  float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(NumDiffs<float>, 1, out, 0, max_d);
}
void GroupedArrayFloat32_NumSeasDiffs(const float *data, const indptr_t *indptr,
                                      int n_indptr, int num_threads, int period,
                                      int max_d, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(NumSeasDiffs<float>, 1, out, 0, period, max_d);
}
void GroupedArrayFloat32_NumSeasDiffsPeriods(const float *data,
                                             const indptr_t *indptr,
                                             int n_indptr, int num_threads,
                                             int max_d,
                                             float *periods_and_out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(NumSeasDiffsPeriods<float>, 2, periods_and_out, 0, max_d);
}

void GroupedArrayFloat32_Period(const float *data, const indptr_t *indptr,
                                int n_indptr, int num_threads, size_t max_lag,
                                float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Reduce(GreatestAutocovariance<float>, 1, out, 0, max_lag);
}
void GroupedArrayFloat32_Difference(const float *data, const indptr_t *indptr,
                                    int n_indptr, int num_threads, int d,
                                    float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.Transform(Difference<float>, 0, out, d);
}
void GroupedArrayFloat32_Differences(const float *data, const indptr_t *indptr,
                                     int n_indptr, int num_threads,
                                     const indptr_t *ds, float *out) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  ga.VariableTransform(Differences<float>, ds, out);
}
void GroupedArrayFloat32_InvertDifferences(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    const float *other_data, const indptr_t *other_indptr, int other_n_indptr,
    const indptr_t *out_indptr, float *out_data) {
  auto ga = GroupedArray<float>(data, indptr, n_indptr, num_threads);
  auto tails_ga = GroupedArray<float>(other_data, other_indptr, other_n_indptr,
                                      num_threads);
  ga.Zip(InvertDifference<float>, tails_ga, out_indptr, out_data);
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
void GroupedArrayFloat64_LagTransform(const double *data,
                                      const indptr_t *indptr, int n_indptr,
                                      int num_threads, int lag, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Transform(LagTransform<double>, lag, out);
}

// Manipulation
void GroupedArrayFloat64_IndexFromEnd(const double *data,
                                      const indptr_t *indptr, int n_indptr,
                                      int num_threads, int k, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(IndexFromEnd<double>, 1, out, 0, k);
}

void GroupedArrayFloat64_Head(const double *data, const indptr_t *indptr,
                              int n_indptr, int num_threads, int k,
                              double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(Head<double>, k, out, 0, k);
}
void GroupedArrayFloat64_Tail(const double *data, const indptr_t *indptr,
                              int n_indptr, int num_threads, int k,
                              double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(Tail<double>, k, out, 0, k);
}
void GroupedArrayFloat64_Append(const double *data, const indptr_t *indptr,
                                int n_indptr, int num_threads,
                                const double *other_data,
                                const indptr_t *other_indptr,
                                int other_n_indptr, const indptr_t *out_indptr,
                                double *out_data) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  auto other = GroupedArray<double>(other_data, other_indptr, other_n_indptr,
                                    num_threads);
  ga.Zip(Append<double>, other, out_indptr, out_data);
}
void GroupedArrayFloat64_Tails(const double *data, const indptr_t *indptr,
                               int n_indptr, int num_threads,
                               const indptr_t *indptr_out, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.VariableReduce(Tail<double>, indptr_out, out);
}

// Lag
// Rolling
void GroupedArrayFloat64_RollingMeanTransform(const double *data,
                                              const indptr_t *indptr,
                                              int n_indptr, int num_threads,
                                              int lag, int window_size,
                                              int min_samples, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Transform(RollingMeanTransform<double>, lag, out, window_size,
               min_samples);
}
void GroupedArrayFloat64_RollingStdTransform(const double *data,
                                             const indptr_t *indptr,
                                             int n_indptr, int num_threads,
                                             int lag, int window_size,
                                             int min_samples, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Transform(RollingStdTransform<double>, lag, out, window_size, min_samples);
}
void GroupedArrayFloat64_RollingMinTransform(const double *data,
                                             const indptr_t *indptr,
                                             int n_indptr, int num_threads,
                                             int lag, int window_size,
                                             int min_samples, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Transform(RollingMinTransform<double>, lag, out, window_size, min_samples);
}
void GroupedArrayFloat64_RollingMaxTransform(const double *data,
                                             const indptr_t *indptr,
                                             int n_indptr, int num_threads,
                                             int lag, int window_size,
                                             int min_samples, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Transform(RollingMaxTransform<double>, lag, out, window_size, min_samples);
}
void GroupedArrayFloat64_RollingQuantileTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, double p, int window_size, int min_samples, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Transform(RollingQuantileTransform<double>, lag, out, window_size,
               min_samples, p);
}
void GroupedArrayFloat64_RollingMeanUpdate(const double *data,
                                           const indptr_t *indptr, int n_indptr,
                                           int num_threads, int lag,
                                           int window_size, int min_samples,
                                           double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(RollingMeanUpdate<double>(), 1, out, lag, window_size, min_samples);
}
void GroupedArrayFloat64_RollingStdUpdate(const double *data,
                                          const indptr_t *indptr, int n_indptr,
                                          int num_threads, int lag,
                                          int window_size, int min_samples,
                                          double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(RollingStdUpdate<double>(), 1, out, lag, window_size, min_samples);
}
void GroupedArrayFloat64_RollingMinUpdate(const double *data,
                                          const indptr_t *indptr, int n_indptr,
                                          int num_threads, int lag,
                                          int window_size, int min_samples,
                                          double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(RollingMinUpdate<double>(), 1, out, lag, window_size, min_samples);
}
void GroupedArrayFloat64_RollingMaxUpdate(const double *data,
                                          const indptr_t *indptr, int n_indptr,
                                          int num_threads, int lag,
                                          int window_size, int min_samples,
                                          double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(RollingMaxUpdate<double>(), 1, out, lag, window_size, min_samples);
}
void GroupedArrayFloat64_RollingQuantileUpdate(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, double p, int window_size, int min_samples, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(RollingQuantileUpdate<double>(), 1, out, lag, window_size,
            min_samples, p);
}

// Seasonal rolling
void GroupedArrayFloat64_SeasonalRollingMeanTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Transform(SeasonalRollingMeanTransform<double>(), lag, out, season_length,
               window_size, min_samples);
}
void GroupedArrayFloat64_SeasonalRollingStdTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Transform(SeasonalRollingStdTransform<double>(), lag, out, season_length,
               window_size, min_samples);
}
void GroupedArrayFloat64_SeasonalRollingMinTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Transform(SeasonalRollingMinTransform<double>(), lag, out, season_length,
               window_size, min_samples);
}
void GroupedArrayFloat64_SeasonalRollingMaxTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Transform(SeasonalRollingMaxTransform<double>(), lag, out, season_length,
               window_size, min_samples);
}
void GroupedArrayFloat64_SeasonalRollingQuantileTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, double p, int window_size, int min_samples,
    double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Transform(SeasonalRollingQuantileTransform<double>(), lag, out,
               season_length, window_size, min_samples, p);
}
void GroupedArrayFloat64_SeasonalRollingMeanUpdate(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(SeasonalRollingMeanUpdate<double>(), 1, out, lag, season_length,
            window_size, min_samples);
}
void GroupedArrayFloat64_SeasonalRollingStdUpdate(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(SeasonalRollingStdUpdate<double>(), 1, out, lag, season_length,
            window_size, min_samples);
}
void GroupedArrayFloat64_SeasonalRollingMinUpdate(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(SeasonalRollingMinUpdate<double>(), 1, out, lag, season_length,
            window_size, min_samples);
}
void GroupedArrayFloat64_SeasonalRollingMaxUpdate(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(SeasonalRollingMaxUpdate<double>(), 1, out, lag, season_length,
            window_size, min_samples);
}
void GroupedArrayFloat64_SeasonalRollingQuantileUpdate(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, double p, int window_size, int min_samples,
    double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(SeasonalRollingQuantileUpdate<double>(), 1, out, lag, season_length,
            window_size, min_samples, p);
}

// Expanding
void GroupedArrayFloat64_ExpandingMeanTransform(const double *data,
                                                const indptr_t *indptr,
                                                int n_indptr, int num_threads,
                                                int lag, double *out,
                                                double *agg) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.TransformAndReduce(ExpandingMeanTransform<double>, lag, out, 1, agg);
}
void GroupedArrayFloat64_ExpandingStdTransform(const double *data,
                                               const indptr_t *indptr,
                                               int n_indptr, int num_threads,
                                               int lag, double *out,
                                               double *agg) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.TransformAndReduce(ExpandingStdTransform<double>, lag, out, 3, agg);
}
void GroupedArrayFloat64_ExpandingMinTransform(const double *data,
                                               const indptr_t *indptr,
                                               int n_indptr, int num_threads,
                                               int lag, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Transform(ExpandingMinTransform<double>(), lag, out);
}
void GroupedArrayFloat64_ExpandingMaxTransform(const double *data,
                                               const indptr_t *indptr,
                                               int n_indptr, int num_threads,
                                               int lag, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Transform(ExpandingMaxTransform<double>(), lag, out);
}
void GroupedArrayFloat64_ExpandingQuantileTransform(const double *data,
                                                    const indptr_t *indptr,
                                                    int n_indptr,
                                                    int num_threads, int lag,
                                                    double p, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Transform(ExpandingQuantileTransform<double>, lag, out, p);
}
void GroupedArrayFloat64_ExpandingQuantileUpdate(const double *data,
                                                 const indptr_t *indptr,
                                                 int n_indptr, int num_threads,
                                                 int lag, double p,
                                                 double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(ExpandingQuantileUpdate<double>, 1, out, lag, p);
}

// Exponentially weighted
void GroupedArrayFloat64_ExponentiallyWeightedMeanTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, double alpha, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Transform(ExponentiallyWeightedMeanTransform<double>, lag, out, alpha);
}

// Scalers
void GroupedArrayFloat64_MinMaxScalerStats(const double *data,
                                           const indptr_t *indptr, int n_indptr,
                                           int num_threads, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(MinMaxScalerStats<double>, 2, out, 0);
}
void GroupedArrayFloat64_StandardScalerStats(const double *data,
                                             const indptr_t *indptr,
                                             int n_indptr, int num_threads,
                                             double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(StandardScalerStats<double>, 2, out, 0);
}
void GroupedArrayFloat64_RobustIqrScalerStats(const double *data,
                                              const indptr_t *indptr,
                                              int n_indptr, int num_threads,
                                              double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(RobustScalerIqrStats<double>, 2, out, 0);
}
void GroupedArrayFloat64_RobustMadScalerStats(const double *data,
                                              const indptr_t *indptr,
                                              int n_indptr, int num_threads,
                                              double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(RobustScalerMadStats<double>, 2, out, 0);
}
void GroupedArrayFloat64_ScalerTransform(const double *data,
                                         const indptr_t *indptr, int n_indptr,
                                         int num_threads, const double *stats,
                                         double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.ScalerTransform(CommonScalerTransform<double>, stats, out);
}
void GroupedArrayFloat64_ScalerInverseTransform(const double *data,
                                                const indptr_t *indptr,
                                                int n_indptr, int num_threads,
                                                const double *stats,
                                                double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.ScalerTransform(CommonScalerInverseTransform<double>, stats, out);
}
void GroupedArrayFloat64_BoxCoxLambdaGuerrero(const double *data,
                                              const indptr_t *indptr,
                                              int n_indptr, int num_threads,
                                              int period, double lower,
                                              double upper, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(BoxCoxLambda_Guerrero<double>, 2, out, 0, period, lower, upper);
}
void GroupedArrayFloat64_BoxCoxLambdaLogLik(const double *data,
                                            const indptr_t *indptr,
                                            int n_indptr, int num_threads,
                                            double lower, double upper,
                                            double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(BoxCoxLambda_LogLik<double>, 2, out, 0, lower, upper);
}
void GroupedArrayFloat64_BoxCoxTransform(const double *data,
                                         const indptr_t *indptr, int n_indptr,
                                         int num_threads, const double *lambdas,
                                         double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.ScalerTransform(BoxCoxTransform<double>, lambdas, out);
}
void GroupedArrayFloat64_BoxCoxInverseTransform(const double *data,
                                                const indptr_t *indptr,
                                                int n_indptr, int num_threads,
                                                const double *lambdas,
                                                double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.ScalerTransform(BoxCoxInverseTransform<double>, lambdas, out);
}

// Differences
void GroupedArrayFloat64_NumDiffs(const double *data, const indptr_t *indptr,
                                  int n_indptr, int num_threads, int max_d,
                                  double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(NumDiffs<double>, 1, out, 0, max_d);
}
void GroupedArrayFloat64_NumSeasDiffs(const double *data,
                                      const indptr_t *indptr, int n_indptr,
                                      int num_threads, int period, int max_d,
                                      double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(NumSeasDiffs<double>, 1, out, 0, period, max_d);
}
void GroupedArrayFloat64_NumSeasDiffsPeriods(const double *data,
                                             const indptr_t *indptr,
                                             int n_indptr, int num_threads,
                                             int max_d,
                                             double *periods_and_out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(NumSeasDiffsPeriods<double>, 2, periods_and_out, 0, max_d);
}

void GroupedArrayFloat64_Period(const double *data, const indptr_t *indptr,
                                int n_indptr, int num_threads, size_t max_lag,
                                double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Reduce(GreatestAutocovariance<double>, 1, out, 0, max_lag);
}
void GroupedArrayFloat64_Difference(const double *data, const indptr_t *indptr,
                                    int n_indptr, int num_threads, int d,
                                    double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.Transform(Difference<double>, 0, out, d);
}
void GroupedArrayFloat64_Differences(const double *data, const indptr_t *indptr,
                                     int n_indptr, int num_threads,
                                     const indptr_t *ds, double *out) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  ga.VariableTransform(Differences<double>, ds, out);
}
void GroupedArrayFloat64_InvertDifferences(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    const double *other_data, const indptr_t *other_indptr, int other_n_indptr,
    const indptr_t *out_indptr, double *out_data) {
  auto ga = GroupedArray<double>(data, indptr, n_indptr, num_threads);
  auto tails_ga = GroupedArray<double>(other_data, other_indptr, other_n_indptr,
                                       num_threads);
  ga.Zip(InvertDifference<double>, tails_ga, out_indptr, out_data);
}
