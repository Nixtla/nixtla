#include "diff.h"
#include "expanding.h"
#include "exponentially_weighted.h"
#include "grouped_array.h"
#include "grouped_array_functions.h"
#include "lag.h"
#include "rolling.h"
#include "scalers.h"

#ifdef _MSC_VER
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

extern "C" {
// Float32 Methods
// Single array
// Rolling
DLL_EXPORT int Float32_RollingMeanTransform(float *data, int length,
                                            int window_size, int min_samples,
                                            float *out);
DLL_EXPORT int Float32_RollingStdTransform(float *data, int length,
                                           int window_size, int min_samples,
                                           float *out);
DLL_EXPORT int Float32_RollingMinTransform(float *data, int length,
                                           int window_size, int min_samples,
                                           float *out);
DLL_EXPORT int Float32_RollingMaxTransform(float *data, int length,
                                           int window_size, int min_samples,
                                           float *out);
DLL_EXPORT int Float32_RollingQuantileTransform(float *data, int length,
                                                float p, int window_size,
                                                int min_samples, float *out);

// Seasonal rolling
DLL_EXPORT int Float32_SeasonalRollingMeanTransform(float *data, int length,
                                                    int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    float *out);
DLL_EXPORT int Float32_SeasonalRollingStdTransform(float *data, int length,
                                                   int season_length,
                                                   int window_size,
                                                   int min_samples, float *out);
DLL_EXPORT int Float32_SeasonalRollingMinTransform(float *data, int length,
                                                   int season_length,
                                                   int window_size,
                                                   int min_samples, float *out);
DLL_EXPORT int Float32_SeasonalRollingMaxTransform(float *data, int length,
                                                   int season_length,
                                                   int window_size,
                                                   int min_samples, float *out);
DLL_EXPORT int Float32_SeasonalRollingQuantileTransform(
    float *data, int length, int season_length, float p, int window_size,
    int min_samples, float *out);

// Expanding
DLL_EXPORT int Float32_ExpandingMeanTransform(float *data, int length,
                                              float *out);
DLL_EXPORT int Float32_ExpandingStdTransform(float *data, int length,
                                             float *out);
DLL_EXPORT int Float32_ExpandingMinTransform(float *data, int length,
                                             float *out);
DLL_EXPORT int Float32_ExpandingMaxTransform(float *data, int length,
                                             float *out);
DLL_EXPORT int Float32_ExpandingQuantileTransform(float *data, int length,
                                                  float p, float *out);
// Exponentially weighted
DLL_EXPORT int Float32_ExponentiallyWeightedMeanTransform(float *data,
                                                          int length,
                                                          float alpha,
                                                          float *out);

// Scalers
DLL_EXPORT float Float32_BoxCoxLambdaGuerrero(const float *x, int n, int period,
                                              float lower, float upper);
DLL_EXPORT float Float32_BoxCoxLambdaLogLik(const float *x, int n, float lower,
                                            float upper);
DLL_EXPORT void Float32_BoxCoxTransform(const float *x, int n, float lambda,
                                        float *out);
DLL_EXPORT void Float32_BoxCoxInverseTransform(const float *x, int n,
                                               float lambda, float *out);

// Differences
DLL_EXPORT void Float32_Difference(const float *x, indptr_t n, int d,
                                   float *out);
DLL_EXPORT int Float32_NumDiffs(const float *x, indptr_t n, int max_d);
DLL_EXPORT int Float32_NumSeasDiffs(const float *x, indptr_t n, int period,
                                    int max_d);
DLL_EXPORT int Float32_Period(const float *x, size_t n, int period);

// GA
// Manipulation
DLL_EXPORT void GroupedArrayFloat32_IndexFromEnd(const float *data,
                                                 const indptr_t *indptr,
                                                 int n_indptr, int num_threads,
                                                 int k, float *out);

// Lag
DLL_EXPORT void GroupedArrayFloat32_LagTransform(const float *data,
                                                 const indptr_t *indptr,
                                                 int n_indptr, int num_threads,
                                                 int lag, float *out);
DLL_EXPORT void GroupedArrayFloat32_IndexFromEnd(const float *data,
                                                 const indptr_t *indptr,
                                                 int n_indptr, int num_threads,
                                                 int k, float *out);
DLL_EXPORT void GroupedArrayFloat32_Head(const float *data,
                                         const indptr_t *indptr, int n_indptr,
                                         int num_threads, int k, float *out);
DLL_EXPORT void GroupedArrayFloat32_Tail(const float *data,
                                         const indptr_t *indptr, int n_indptr,
                                         int num_threads, int k, float *out);
DLL_EXPORT void GroupedArrayFloat32_Append(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    const float *other_data, const indptr_t *other_indptr, int other_n_indptr,
    const indptr_t *out_indptr, float *out_data);
DLL_EXPORT void GroupedArrayFloat32_Tails(const float *data,
                                          const indptr_t *indptr, int n_indptr,
                                          int num_threads,
                                          const indptr_t *indptr_out,
                                          float *out);

// Rolling
DLL_EXPORT void GroupedArrayFloat32_RollingMeanTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int window_size, int min_samples, float *out);
DLL_EXPORT void GroupedArrayFloat32_RollingStdTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int window_size, int min_samples, float *out);
DLL_EXPORT void GroupedArrayFloat32_RollingMinTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int window_size, int min_samples, float *out);
DLL_EXPORT void GroupedArrayFloat32_RollingMaxTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int window_size, int min_samples, float *out);
DLL_EXPORT void GroupedArrayFloat32_RollingQuantileTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, float p, int window_size, int min_samples, float *out);
DLL_EXPORT void GroupedArrayFloat32_RollingMeanUpdate(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int window_size, int min_samples, float *out);
DLL_EXPORT void GroupedArrayFloat32_RollingStdUpdate(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int window_size, int min_samples, float *out);
DLL_EXPORT void GroupedArrayFloat32_RollingMinUpdate(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int window_size, int min_samples, float *out);
DLL_EXPORT void GroupedArrayFloat32_RollingMaxUpdate(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int window_size, int min_samples, float *out);
DLL_EXPORT void GroupedArrayFloat32_RollingQuantileUpdate(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, float p, int window_size, int min_samples, float *out);

// Seasonal rolling
DLL_EXPORT void GroupedArrayFloat32_SeasonalRollingMeanTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, float *out);
DLL_EXPORT void GroupedArrayFloat32_SeasonalRollingStdTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, float *out);
DLL_EXPORT void GroupedArrayFloat32_SeasonalRollingMinTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, float *out);
DLL_EXPORT void GroupedArrayFloat32_SeasonalRollingMaxTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, float *out);
DLL_EXPORT void GroupedArrayFloat32_SeasonalRollingQuantileTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, float p, int window_size, int min_samples,
    float *out);
DLL_EXPORT void GroupedArrayFloat32_SeasonalRollingMeanUpdate(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, float *out);
DLL_EXPORT void GroupedArrayFloat32_SeasonalRollingStdUpdate(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, float *out);
DLL_EXPORT void GroupedArrayFloat32_SeasonalRollingMinUpdate(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, float *out);
DLL_EXPORT void GroupedArrayFloat32_SeasonalRollingMaxUpdate(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, float *out);
DLL_EXPORT void GroupedArrayFloat32_SeasonalRollingQuantileUpdate(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, float p, int window_size, int min_samples,
    float *out);

// Expanding
DLL_EXPORT void GroupedArrayFloat32_ExpandingMeanTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, float *out, float *agg);
DLL_EXPORT void GroupedArrayFloat32_ExpandingStdTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, float *out, float *agg);
DLL_EXPORT void
GroupedArrayFloat32_ExpandingMinTransform(const float *data,
                                          const indptr_t *indptr, int n_indptr,
                                          int num_threads, int lag, float *out);
DLL_EXPORT void
GroupedArrayFloat32_ExpandingMaxTransform(const float *data,
                                          const indptr_t *indptr, int n_indptr,
                                          int num_threads, int lag, float *out);
DLL_EXPORT void GroupedArrayFloat32_ExpandingQuantileTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, float p, float *out);
DLL_EXPORT void GroupedArrayFloat32_ExpandingQuantileUpdate(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, float p, float *out);

// Exponentially weighted
DLL_EXPORT void GroupedArrayFloat32_ExponentiallyWeightedMeanTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, float alpha, float *out);

// Scalers
DLL_EXPORT void GroupedArrayFloat32_MinMaxScalerStats(const float *data,
                                                      const indptr_t *indptr,
                                                      int n_indptr,
                                                      int num_threads,
                                                      float *out);
DLL_EXPORT void GroupedArrayFloat32_StandardScalerStats(const float *data,
                                                        const indptr_t *indptr,
                                                        int n_indptr,
                                                        int num_threads,
                                                        float *out);
DLL_EXPORT void GroupedArrayFloat32_RobustIqrScalerStats(const float *data,
                                                         const indptr_t *indptr,
                                                         int n_indptr,
                                                         int num_threads,
                                                         float *out);
DLL_EXPORT void GroupedArrayFloat32_RobustMadScalerStats(const float *data,
                                                         const indptr_t *indptr,
                                                         int n_indptr,
                                                         int num_threads,
                                                         float *out);
DLL_EXPORT void
GroupedArrayFloat32_ScalerTransform(const float *data, const indptr_t *indptr,
                                    int n_indptr, int num_threads,
                                    const float *stats, float *out);
DLL_EXPORT void GroupedArrayFloat32_ScalerInverseTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    const float *stats, float *out);
DLL_EXPORT void GroupedArrayFloat32_BoxCoxLambdaGuerrero(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int period, float lower, float upper, float *out);
DLL_EXPORT void GroupedArrayFloat32_BoxCoxLambdaLogLik(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    float lower, float upper, float *out);
DLL_EXPORT void
GroupedArrayFloat32_BoxCoxTransform(const float *data, const indptr_t *indptr,
                                    int n_indptr, int num_threads,
                                    const float *lambdas, float *out);
DLL_EXPORT void GroupedArrayFloat32_BoxCoxInverseTransform(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    const float *lambdas, float *out);

// Differences
DLL_EXPORT void GroupedArrayFloat32_NumDiffs(const float *data,
                                             const indptr_t *indptr,
                                             int n_indptr, int num_threads,
                                             int max_d, float *out);
DLL_EXPORT void GroupedArrayFloat32_NumSeasDiffs(const float *data,
                                                 const indptr_t *indptr,
                                                 int n_indptr, int num_threads,
                                                 int period, int max_d,
                                                 float *out);
DLL_EXPORT void GroupedArrayFloat32_NumSeasDiffsPeriods(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int max_d, float *periods_and_out);
DLL_EXPORT void GroupedArrayFloat32_Period(const float *data,
                                           const indptr_t *indptr, int n_indptr,
                                           int num_threads, size_t max_lag,
                                           float *out);
DLL_EXPORT void GroupedArrayFloat32_Difference(const float *data,
                                               const indptr_t *indptr,
                                               int n_indptr, int num_threads,
                                               int d, float *out);
DLL_EXPORT void GroupedArrayFloat32_Differences(const float *data,
                                                const indptr_t *indptr,
                                                int n_indptr, int num_threads,
                                                const indptr_t *ds, float *out);
DLL_EXPORT void GroupedArrayFloat32_InvertDifferences(
    const float *data, const indptr_t *indptr, int n_indptr, int num_threads,
    const float *other_data, const indptr_t *other_indptr, int other_n_indptr,
    const indptr_t *out_indptr, float *out_data);

// Float64 Methods
//  Single array
//  Rolling
DLL_EXPORT int Float64_RollingMeanTransform(double *data, int length,
                                            int window_size, int min_samples,
                                            double *out);
DLL_EXPORT int Float64_RollingStdTransform(double *data, int length,
                                           int window_size, int min_samples,
                                           double *out);
DLL_EXPORT int Float64_RollingMinTransform(double *data, int length,
                                           int window_size, int min_samples,
                                           double *out);
DLL_EXPORT int Float64_RollingMaxTransform(double *data, int length,
                                           int window_size, int min_samples,
                                           double *out);
DLL_EXPORT int Float64_RollingQuantileTransform(double *data, int length,
                                                double p, int window_size,
                                                int min_samples, double *out);

// Seasonal rolling
DLL_EXPORT int Float64_SeasonalRollingMeanTransform(double *data, int length,
                                                    int season_length,
                                                    int window_size,
                                                    int min_samples,
                                                    double *out);
DLL_EXPORT int Float64_SeasonalRollingStdTransform(double *data, int length,
                                                   int season_length,
                                                   int window_size,
                                                   int min_samples,
                                                   double *out);
DLL_EXPORT int Float64_SeasonalRollingMinTransform(double *data, int length,
                                                   int season_length,
                                                   int window_size,
                                                   int min_samples,
                                                   double *out);
DLL_EXPORT int Float64_SeasonalRollingMaxTransform(double *data, int length,
                                                   int season_length,
                                                   int window_size,
                                                   int min_samples,
                                                   double *out);
DLL_EXPORT int Float64_SeasonalRollingQuantileTransform(
    double *data, int length, int season_length, double p, int window_size,
    int min_samples, double *out);

// Expanding
DLL_EXPORT int Float64_ExpandingMeanTransform(double *data, int length,
                                              double *out);
DLL_EXPORT int Float64_ExpandingStdTransform(double *data, int length,
                                             double *out);
DLL_EXPORT int Float64_ExpandingMinTransform(double *data, int length,
                                             double *out);
DLL_EXPORT int Float64_ExpandingMaxTransform(double *data, int length,
                                             double *out);
DLL_EXPORT int Float64_ExpandingQuantileTransform(double *data, int length,
                                                  double p, double *out);
// Exponentially weighted
DLL_EXPORT int Float64_ExponentiallyWeightedMeanTransform(double *data,
                                                          int length,
                                                          double alpha,
                                                          double *out);

// Scalers
DLL_EXPORT double Float64_BoxCoxLambdaGuerrero(const double *x, int n,
                                               int period, double lower,
                                               double upper);
DLL_EXPORT double Float64_BoxCoxLambdaLogLik(const double *x, int n,
                                             double lower, double upper);
DLL_EXPORT void Float64_BoxCoxTransform(const double *x, int n, double lambda,
                                        double *out);
DLL_EXPORT void Float64_BoxCoxInverseTransform(const double *x, int n,
                                               double lambda, double *out);

// Differences
DLL_EXPORT void Float64_Difference(const double *x, indptr_t n, int d,
                                   double *out);
DLL_EXPORT int Float64_NumDiffs(const double *x, indptr_t n, int max_d);
DLL_EXPORT int Float64_NumSeasDiffs(const double *x, indptr_t n, int period,
                                    int max_d);
DLL_EXPORT int Float64_Period(const double *x, size_t n, int period);

// GA
// Manipulation
DLL_EXPORT void GroupedArrayFloat64_IndexFromEnd(const double *data,
                                                 const indptr_t *indptr,
                                                 int n_indptr, int num_threads,
                                                 int k, double *out);

// Lag
DLL_EXPORT void GroupedArrayFloat64_LagTransform(const double *data,
                                                 const indptr_t *indptr,
                                                 int n_indptr, int num_threads,
                                                 int lag, double *out);
DLL_EXPORT void GroupedArrayFloat64_IndexFromEnd(const double *data,
                                                 const indptr_t *indptr,
                                                 int n_indptr, int num_threads,
                                                 int k, double *out);
DLL_EXPORT void GroupedArrayFloat64_Head(const double *data,
                                         const indptr_t *indptr, int n_indptr,
                                         int num_threads, int k, double *out);
DLL_EXPORT void GroupedArrayFloat64_Tail(const double *data,
                                         const indptr_t *indptr, int n_indptr,
                                         int num_threads, int k, double *out);
DLL_EXPORT void GroupedArrayFloat64_Append(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    const double *other_data, const indptr_t *other_indptr, int other_n_indptr,
    const indptr_t *out_indptr, double *out_data);
DLL_EXPORT void GroupedArrayFloat64_Tails(const double *data,
                                          const indptr_t *indptr, int n_indptr,
                                          int num_threads,
                                          const indptr_t *indptr_out,
                                          double *out);

// Rolling
DLL_EXPORT void GroupedArrayFloat64_RollingMeanTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int window_size, int min_samples, double *out);
DLL_EXPORT void GroupedArrayFloat64_RollingStdTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int window_size, int min_samples, double *out);
DLL_EXPORT void GroupedArrayFloat64_RollingMinTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int window_size, int min_samples, double *out);
DLL_EXPORT void GroupedArrayFloat64_RollingMaxTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int window_size, int min_samples, double *out);
DLL_EXPORT void GroupedArrayFloat64_RollingQuantileTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, double p, int window_size, int min_samples, double *out);
DLL_EXPORT void GroupedArrayFloat64_RollingMeanUpdate(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int window_size, int min_samples, double *out);
DLL_EXPORT void GroupedArrayFloat64_RollingStdUpdate(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int window_size, int min_samples, double *out);
DLL_EXPORT void GroupedArrayFloat64_RollingMinUpdate(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int window_size, int min_samples, double *out);
DLL_EXPORT void GroupedArrayFloat64_RollingMaxUpdate(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int window_size, int min_samples, double *out);
DLL_EXPORT void GroupedArrayFloat64_RollingQuantileUpdate(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, double p, int window_size, int min_samples, double *out);

// Seasonal rolling
DLL_EXPORT void GroupedArrayFloat64_SeasonalRollingMeanTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, double *out);
DLL_EXPORT void GroupedArrayFloat64_SeasonalRollingStdTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, double *out);
DLL_EXPORT void GroupedArrayFloat64_SeasonalRollingMinTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, double *out);
DLL_EXPORT void GroupedArrayFloat64_SeasonalRollingMaxTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, double *out);
DLL_EXPORT void GroupedArrayFloat64_SeasonalRollingQuantileTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, double p, int window_size, int min_samples,
    double *out);
DLL_EXPORT void GroupedArrayFloat64_SeasonalRollingMeanUpdate(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, double *out);
DLL_EXPORT void GroupedArrayFloat64_SeasonalRollingStdUpdate(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, double *out);
DLL_EXPORT void GroupedArrayFloat64_SeasonalRollingMinUpdate(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, double *out);
DLL_EXPORT void GroupedArrayFloat64_SeasonalRollingMaxUpdate(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, int window_size, int min_samples, double *out);
DLL_EXPORT void GroupedArrayFloat64_SeasonalRollingQuantileUpdate(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, int season_length, double p, int window_size, int min_samples,
    double *out);

// Expanding
DLL_EXPORT void GroupedArrayFloat64_ExpandingMeanTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, double *out, double *agg);
DLL_EXPORT void GroupedArrayFloat64_ExpandingStdTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, double *out, double *agg);
DLL_EXPORT void GroupedArrayFloat64_ExpandingMinTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, double *out);
DLL_EXPORT void GroupedArrayFloat64_ExpandingMaxTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, double *out);
DLL_EXPORT void GroupedArrayFloat64_ExpandingQuantileTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, double p, double *out);
DLL_EXPORT void GroupedArrayFloat64_ExpandingQuantileUpdate(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, double p, double *out);

// Exponentially weighted
DLL_EXPORT void GroupedArrayFloat64_ExponentiallyWeightedMeanTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int lag, double alpha, double *out);

// Scalers
DLL_EXPORT void GroupedArrayFloat64_MinMaxScalerStats(const double *data,
                                                      const indptr_t *indptr,
                                                      int n_indptr,
                                                      int num_threads,
                                                      double *out);
DLL_EXPORT void GroupedArrayFloat64_StandardScalerStats(const double *data,
                                                        const indptr_t *indptr,
                                                        int n_indptr,
                                                        int num_threads,
                                                        double *out);
DLL_EXPORT void GroupedArrayFloat64_RobustIqrScalerStats(const double *data,
                                                         const indptr_t *indptr,
                                                         int n_indptr,
                                                         int num_threads,
                                                         double *out);
DLL_EXPORT void GroupedArrayFloat64_RobustMadScalerStats(const double *data,
                                                         const indptr_t *indptr,
                                                         int n_indptr,
                                                         int num_threads,
                                                         double *out);
DLL_EXPORT void
GroupedArrayFloat64_ScalerTransform(const double *data, const indptr_t *indptr,
                                    int n_indptr, int num_threads,
                                    const double *stats, double *out);
DLL_EXPORT void GroupedArrayFloat64_ScalerInverseTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    const double *stats, double *out);
DLL_EXPORT void GroupedArrayFloat64_BoxCoxLambdaGuerrero(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int period, double lower, double upper, double *out);
DLL_EXPORT void GroupedArrayFloat64_BoxCoxLambdaLogLik(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    double lower, double upper, double *out);
DLL_EXPORT void
GroupedArrayFloat64_BoxCoxTransform(const double *data, const indptr_t *indptr,
                                    int n_indptr, int num_threads,
                                    const double *lambdas, double *out);
DLL_EXPORT void GroupedArrayFloat64_BoxCoxInverseTransform(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    const double *lambdas, double *out);

// Differences
DLL_EXPORT void GroupedArrayFloat64_NumDiffs(const double *data,
                                             const indptr_t *indptr,
                                             int n_indptr, int num_threads,
                                             int max_d, double *out);
DLL_EXPORT void GroupedArrayFloat64_NumSeasDiffs(const double *data,
                                                 const indptr_t *indptr,
                                                 int n_indptr, int num_threads,
                                                 int period, int max_d,
                                                 double *out);
DLL_EXPORT void GroupedArrayFloat64_NumSeasDiffsPeriods(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    int max_d, double *periods_and_out);
DLL_EXPORT void GroupedArrayFloat64_Period(const double *data,
                                           const indptr_t *indptr, int n_indptr,
                                           int num_threads, size_t max_lag,
                                           double *out);
DLL_EXPORT void GroupedArrayFloat64_Difference(const double *data,
                                               const indptr_t *indptr,
                                               int n_indptr, int num_threads,
                                               int d, double *out);
DLL_EXPORT void GroupedArrayFloat64_Differences(const double *data,
                                                const indptr_t *indptr,
                                                int n_indptr, int num_threads,
                                                const indptr_t *ds,
                                                double *out);
DLL_EXPORT void GroupedArrayFloat64_InvertDifferences(
    const double *data, const indptr_t *indptr, int n_indptr, int num_threads,
    const double *other_data, const indptr_t *other_indptr, int other_n_indptr,
    const indptr_t *out_indptr, double *out_data);
}
