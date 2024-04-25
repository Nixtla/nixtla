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
DLL_EXPORT int GroupedArrayFloat32_Create(const float *data, indptr_t n_data,
                                          indptr_t *indptr, indptr_t n_indptr,
                                          int num_threads,
                                          GroupedArrayHandle *out);
DLL_EXPORT int GroupedArrayFloat32_Delete(GroupedArrayHandle handle);

DLL_EXPORT int GroupedArrayFloat32_IndexFromEnd(GroupedArrayHandle handle,
                                                int k, float *out);
DLL_EXPORT void GroupedArrayFloat32_Head(GroupedArrayHandle handle, int k,
                                         float *out);
DLL_EXPORT void GroupedArrayFloat32_Tail(GroupedArrayHandle handle, int k,
                                         float *out);
DLL_EXPORT void GroupedArrayFloat32_Tails(GroupedArrayHandle handle,
                                          const indptr_t *indptr_out,
                                          float *out);
DLL_EXPORT void GroupedArrayFloat32_Append(GroupedArrayHandle handle,
                                           GroupedArrayHandle other_handle,
                                           const indptr_t *out_indptr,
                                           float *out_data);
// Lag
DLL_EXPORT int GroupedArrayFloat32_LagTransform(GroupedArrayHandle handle,
                                                int lag, float *out);

// Rolling
DLL_EXPORT int
GroupedArrayFloat32_RollingMeanTransform(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         float *out);
DLL_EXPORT int
GroupedArrayFloat32_RollingStdTransform(GroupedArrayHandle handle, int lag,
                                        int window_size, int min_samples,
                                        float *out);
DLL_EXPORT int
GroupedArrayFloat32_RollingMinTransform(GroupedArrayHandle handle, int lag,
                                        int window_size, int min_samples,
                                        float *out);
DLL_EXPORT int
GroupedArrayFloat32_RollingMaxTransform(GroupedArrayHandle handle, int lag,
                                        int window_size, int min_samples,
                                        float *out);
DLL_EXPORT int
GroupedArrayFloat32_RollingQuantileTransform(GroupedArrayHandle handle, int lag,
                                             float p, int window_size,
                                             int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat32_RollingMeanUpdate(GroupedArrayHandle handle,
                                                     int lag, int window_size,
                                                     int min_samples,
                                                     float *out);
DLL_EXPORT int GroupedArrayFloat32_RollingStdUpdate(GroupedArrayHandle handle,
                                                    int lag, int window_size,
                                                    int min_samples,
                                                    float *out);
DLL_EXPORT int GroupedArrayFloat32_RollingMinUpdate(GroupedArrayHandle handle,
                                                    int lag, int window_size,
                                                    int min_samples,
                                                    float *out);
DLL_EXPORT int GroupedArrayFloat32_RollingMaxUpdate(GroupedArrayHandle handle,
                                                    int lag, int window_size,
                                                    int min_samples,
                                                    float *out);
DLL_EXPORT int
GroupedArrayFloat32_RollingQuantileUpdate(GroupedArrayHandle handle, int lag,
                                          float p, int window_size,
                                          int min_samples, float *out);

// Seasonal rolling
DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingMeanTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingStdTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingMinTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingMaxTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingQuantileTransform(
    GroupedArrayHandle handle, int lag, int season_length, float p,
    int window_size, int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingMeanUpdate(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, float *out);
DLL_EXPORT int
GroupedArrayFloat32_SeasonalRollingStdUpdate(GroupedArrayHandle handle, int lag,
                                             int season_length, int window_size,
                                             int min_samples, float *out);
DLL_EXPORT int
GroupedArrayFloat32_SeasonalRollingMinUpdate(GroupedArrayHandle handle, int lag,
                                             int season_length, int window_size,
                                             int min_samples, float *out);
DLL_EXPORT int
GroupedArrayFloat32_SeasonalRollingMaxUpdate(GroupedArrayHandle handle, int lag,
                                             int season_length, int window_size,
                                             int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingQuantileUpdate(
    GroupedArrayHandle handle, int lag, int season_length, float p,
    int window_size, int min_samples, float *out);

// Expanding
DLL_EXPORT int
GroupedArrayFloat32_ExpandingMeanTransform(GroupedArrayHandle handle, int lag,
                                           float *out, float *agg);
DLL_EXPORT int
GroupedArrayFloat32_ExpandingStdTransform(GroupedArrayHandle handle, int lag,
                                          float *out, float *agg);
DLL_EXPORT int
GroupedArrayFloat32_ExpandingMinTransform(GroupedArrayHandle handle, int lag,
                                          float *out);
DLL_EXPORT int
GroupedArrayFloat32_ExpandingMaxTransform(GroupedArrayHandle handle, int lag,
                                          float *out);
DLL_EXPORT int
GroupedArrayFloat32_ExpandingQuantileTransform(GroupedArrayHandle handle,
                                               int lag, float p, float *out);
DLL_EXPORT int
GroupedArrayFloat32_ExpandingQuantileUpdate(GroupedArrayHandle handle, int lag,
                                            float p, float *out);

// Exponentially weighted
DLL_EXPORT int GroupedArrayFloat32_ExponentiallyWeightedMeanTransform(
    GroupedArrayHandle handle, int lag, float alpha, float *out);

// Scalers
DLL_EXPORT int GroupedArrayFloat32_MinMaxScalerStats(GroupedArrayHandle handle,
                                                     float *out);
DLL_EXPORT int
GroupedArrayFloat32_StandardScalerStats(GroupedArrayHandle handle, float *out);

DLL_EXPORT int
GroupedArrayFloat32_RobustIqrScalerStats(GroupedArrayHandle handle, float *out);

DLL_EXPORT int
GroupedArrayFloat32_RobustMadScalerStats(GroupedArrayHandle handle, float *out);

DLL_EXPORT int GroupedArrayFloat32_ScalerTransform(GroupedArrayHandle handle,
                                                   const float *stats,
                                                   float *out);
DLL_EXPORT int
GroupedArrayFloat32_ScalerInverseTransform(GroupedArrayHandle handle,
                                           const float *stats, float *out);
DLL_EXPORT int
GroupedArrayFloat32_BoxCoxLambdaGuerrero(GroupedArrayHandle handle, int period,
                                         float lower, float upper, float *out);

DLL_EXPORT void
GroupedArrayFloat32_BoxCoxLambdaLogLik(GroupedArrayHandle handle, float lower,
                                       float upper, float *out);
DLL_EXPORT int GroupedArrayFloat32_BoxCoxTransform(GroupedArrayHandle handle,
                                                   const float *lambdas,
                                                   float *out);
DLL_EXPORT int
GroupedArrayFloat32_BoxCoxInverseTransform(GroupedArrayHandle handle,
                                           const float *lambdas, float *out);

// Differences
DLL_EXPORT void GroupedArrayFloat32_NumDiffs(GroupedArrayHandle handle,
                                             int max_d, float *out);
DLL_EXPORT void GroupedArrayFloat32_NumSeasDiffs(GroupedArrayHandle handle,
                                                 int period, int max_d,
                                                 float *out);
DLL_EXPORT void
GroupedArrayFloat32_NumSeasDiffsPeriods(GroupedArrayHandle handle, int max_d,
                                        float *periods_and_out);
DLL_EXPORT void GroupedArrayFloat32_Period(GroupedArrayHandle handle,
                                           size_t max_lag, float *out);
DLL_EXPORT void GroupedArrayFloat32_Difference(GroupedArrayHandle handle, int d,
                                               float *out);
DLL_EXPORT void GroupedArrayFloat32_Differences(GroupedArrayHandle handle,
                                                const indptr_t *ds, float *out);
DLL_EXPORT void GroupedArrayFloat32_InvertDifferences(
    GroupedArrayHandle handle, GroupedArrayHandle tails_handle,
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
DLL_EXPORT int GroupedArrayFloat64_Create(const double *data, indptr_t n_data,
                                          indptr_t *indptr, indptr_t n_indptr,
                                          int num_threads,
                                          GroupedArrayHandle *out);
DLL_EXPORT int GroupedArrayFloat64_Delete(GroupedArrayHandle handle);

DLL_EXPORT int GroupedArrayFloat64_IndexFromEnd(GroupedArrayHandle handle,
                                                int k, double *out);
DLL_EXPORT void GroupedArrayFloat64_Head(GroupedArrayHandle handle, int k,
                                         double *out);
DLL_EXPORT void GroupedArrayFloat64_Tail(GroupedArrayHandle handle, int k,
                                         double *out);
DLL_EXPORT void GroupedArrayFloat64_Tails(GroupedArrayHandle handle,
                                          const indptr_t *indptr_out,
                                          double *out);
DLL_EXPORT void GroupedArrayFloat64_Append(GroupedArrayHandle handle,
                                           GroupedArrayHandle other_handle,
                                           const indptr_t *out_indptr,
                                           double *out_data);
// Lag
DLL_EXPORT int GroupedArrayFloat64_LagTransform(GroupedArrayHandle handle,
                                                int lag, double *out);

// Rolling
DLL_EXPORT int
GroupedArrayFloat64_RollingMeanTransform(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         double *out);
DLL_EXPORT int
GroupedArrayFloat64_RollingStdTransform(GroupedArrayHandle handle, int lag,
                                        int window_size, int min_samples,
                                        double *out);
DLL_EXPORT int
GroupedArrayFloat64_RollingMinTransform(GroupedArrayHandle handle, int lag,
                                        int window_size, int min_samples,
                                        double *out);
DLL_EXPORT int
GroupedArrayFloat64_RollingMaxTransform(GroupedArrayHandle handle, int lag,
                                        int window_size, int min_samples,
                                        double *out);
DLL_EXPORT int
GroupedArrayFloat64_RollingQuantileTransform(GroupedArrayHandle handle, int lag,
                                             double p, int window_size,
                                             int min_samples, double *out);
DLL_EXPORT int GroupedArrayFloat64_RollingMeanUpdate(GroupedArrayHandle handle,
                                                     int lag, int window_size,
                                                     int min_samples,
                                                     double *out);
DLL_EXPORT int GroupedArrayFloat64_RollingStdUpdate(GroupedArrayHandle handle,
                                                    int lag, int window_size,
                                                    int min_samples,
                                                    double *out);
DLL_EXPORT int GroupedArrayFloat64_RollingMinUpdate(GroupedArrayHandle handle,
                                                    int lag, int window_size,
                                                    int min_samples,
                                                    double *out);
DLL_EXPORT int GroupedArrayFloat64_RollingMaxUpdate(GroupedArrayHandle handle,
                                                    int lag, int window_size,
                                                    int min_samples,
                                                    double *out);
DLL_EXPORT int
GroupedArrayFloat64_RollingQuantileUpdate(GroupedArrayHandle handle, int lag,
                                          double p, int window_size,
                                          int min_samples, double *out);

// Seasonal rolling
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingMeanTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, double *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingStdTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, double *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingMinTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, double *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingMaxTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, double *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingQuantileTransform(
    GroupedArrayHandle handle, int lag, int season_length, double p,
    int window_size, int min_samples, double *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingMeanUpdate(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, double *out);
DLL_EXPORT int
GroupedArrayFloat64_SeasonalRollingStdUpdate(GroupedArrayHandle handle, int lag,
                                             int season_length, int window_size,
                                             int min_samples, double *out);
DLL_EXPORT int
GroupedArrayFloat64_SeasonalRollingMinUpdate(GroupedArrayHandle handle, int lag,
                                             int season_length, int window_size,
                                             int min_samples, double *out);
DLL_EXPORT int
GroupedArrayFloat64_SeasonalRollingMaxUpdate(GroupedArrayHandle handle, int lag,
                                             int season_length, int window_size,
                                             int min_samples, double *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingQuantileUpdate(
    GroupedArrayHandle handle, int lag, int season_length, double p,
    int window_size, int min_samples, double *out);

// Expanding
DLL_EXPORT int
GroupedArrayFloat64_ExpandingMeanTransform(GroupedArrayHandle handle, int lag,
                                           double *out, double *agg);
DLL_EXPORT int
GroupedArrayFloat64_ExpandingStdTransform(GroupedArrayHandle handle, int lag,
                                          double *out, double *agg);
DLL_EXPORT int
GroupedArrayFloat64_ExpandingMinTransform(GroupedArrayHandle handle, int lag,
                                          double *out);
DLL_EXPORT int
GroupedArrayFloat64_ExpandingMaxTransform(GroupedArrayHandle handle, int lag,
                                          double *out);
DLL_EXPORT int
GroupedArrayFloat64_ExpandingQuantileTransform(GroupedArrayHandle handle,
                                               int lag, double p, double *out);
DLL_EXPORT int
GroupedArrayFloat64_ExpandingQuantileUpdate(GroupedArrayHandle handle, int lag,
                                            double p, double *out);

// Exponentially weighted
DLL_EXPORT int GroupedArrayFloat64_ExponentiallyWeightedMeanTransform(
    GroupedArrayHandle handle, int lag, double alpha, double *out);

// Scalers
DLL_EXPORT int GroupedArrayFloat64_MinMaxScalerStats(GroupedArrayHandle handle,
                                                     double *out);
DLL_EXPORT int
GroupedArrayFloat64_StandardScalerStats(GroupedArrayHandle handle, double *out);

DLL_EXPORT int
GroupedArrayFloat64_RobustIqrScalerStats(GroupedArrayHandle handle,
                                         double *out);

DLL_EXPORT int
GroupedArrayFloat64_RobustMadScalerStats(GroupedArrayHandle handle,
                                         double *out);

DLL_EXPORT int GroupedArrayFloat64_ScalerTransform(GroupedArrayHandle handle,
                                                   const double *stats,
                                                   double *out);
DLL_EXPORT int
GroupedArrayFloat64_ScalerInverseTransform(GroupedArrayHandle handle,
                                           const double *stats, double *out);
DLL_EXPORT int
GroupedArrayFloat64_BoxCoxLambdaGuerrero(GroupedArrayHandle handle, int period,
                                         double lower, double upper,
                                         double *out);

DLL_EXPORT void
GroupedArrayFloat64_BoxCoxLambdaLogLik(GroupedArrayHandle handle, double lower,
                                       double upper, double *out);
DLL_EXPORT int GroupedArrayFloat64_BoxCoxTransform(GroupedArrayHandle handle,
                                                   const double *lambdas,
                                                   double *out);
DLL_EXPORT int
GroupedArrayFloat64_BoxCoxInverseTransform(GroupedArrayHandle handle,
                                           const double *lambdas, double *out);

// Differences
DLL_EXPORT void GroupedArrayFloat64_NumDiffs(GroupedArrayHandle handle,
                                             int max_d, double *out);
DLL_EXPORT void GroupedArrayFloat64_NumSeasDiffs(GroupedArrayHandle handle,
                                                 int period, int max_d,
                                                 double *out);
DLL_EXPORT void
GroupedArrayFloat64_NumSeasDiffsPeriods(GroupedArrayHandle handle, int max_d,
                                        double *periods_and_out);
DLL_EXPORT void GroupedArrayFloat64_Period(GroupedArrayHandle handle,
                                           size_t max_lag, double *out);
DLL_EXPORT void GroupedArrayFloat64_Difference(GroupedArrayHandle handle, int d,
                                               double *out);
DLL_EXPORT void GroupedArrayFloat64_Differences(GroupedArrayHandle handle,
                                                const indptr_t *ds,
                                                double *out);
DLL_EXPORT void GroupedArrayFloat64_InvertDifferences(
    GroupedArrayHandle handle, GroupedArrayHandle tails_handle,
    const indptr_t *out_indptr, double *out_data);
}
