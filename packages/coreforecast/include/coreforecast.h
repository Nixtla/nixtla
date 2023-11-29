#pragma once

#include <cstdint>

#ifdef _MSC_VER
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

using GroupedArrayHandle = void *;
using indptr_t = int32_t;

extern "C" {
DLL_EXPORT int GroupedArrayFloat32_Create(const float *data, int32_t n_data,
                                          int32_t *indptr, int32_t n_indptr,
                                          int num_threads,
                                          GroupedArrayHandle *out);
DLL_EXPORT int GroupedArrayFloat64_Create(const double *data, int32_t n_data,
                                          int32_t *indptr, int32_t n_indptr,
                                          int num_threads,
                                          GroupedArrayHandle *out);

DLL_EXPORT int GroupedArrayFloat32_Delete(GroupedArrayHandle handle);
DLL_EXPORT int GroupedArrayFloat64_Delete(GroupedArrayHandle handle);

DLL_EXPORT int GroupedArrayFloat32_MinMaxScalerStats(GroupedArrayHandle handle,
                                                     float *out);
DLL_EXPORT int GroupedArrayFloat64_MinMaxScalerStats(GroupedArrayHandle handle,
                                                     double *out);

DLL_EXPORT int
GroupedArrayFloat32_StandardScalerStats(GroupedArrayHandle handle, float *out);
DLL_EXPORT int
GroupedArrayFloat64_StandardScalerStats(GroupedArrayHandle handle, double *out);

DLL_EXPORT int
GroupedArrayFloat32_RobustIqrScalerStats(GroupedArrayHandle handle, float *out);
DLL_EXPORT int
GroupedArrayFloat64_RobustIqrScalerStats(GroupedArrayHandle handle,
                                         double *out);

DLL_EXPORT int
GroupedArrayFloat32_RobustMadScalerStats(GroupedArrayHandle handle, float *out);
DLL_EXPORT int
GroupedArrayFloat64_RobustMadScalerStats(GroupedArrayHandle handle,
                                         double *out);

DLL_EXPORT int GroupedArrayFloat32_ScalerTransform(GroupedArrayHandle handle,
                                                   const float *stats,
                                                   float *out);
DLL_EXPORT int GroupedArrayFloat64_ScalerTransform(GroupedArrayHandle handle,
                                                   const double *stats,
                                                   double *out);

DLL_EXPORT int
GroupedArrayFloat32_ScalerInverseTransform(GroupedArrayHandle handle,
                                           const float *stats, float *out);
DLL_EXPORT int
GroupedArrayFloat64_ScalerInverseTransform(GroupedArrayHandle handle,
                                           const double *stats, double *out);

DLL_EXPORT int GroupedArrayFloat32_TakeFromGroups(GroupedArrayHandle handle,
                                                  int k, float *out);
DLL_EXPORT int GroupedArrayFloat64_TakeFromGroups(GroupedArrayHandle handle,
                                                  int k, double *out);

DLL_EXPORT int GroupedArrayFloat32_LagTransform(GroupedArrayHandle handle,
                                                int lag, float *out);
DLL_EXPORT int GroupedArrayFloat64_LagTransform(GroupedArrayHandle handle,
                                                int lag, double *out);

DLL_EXPORT int
GroupedArrayFloat32_RollingMeanTransform(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         float *out);
DLL_EXPORT int
GroupedArrayFloat64_RollingMeanTransform(GroupedArrayHandle handle, int lag,
                                         int window_size, int min_samples,
                                         double *out);

DLL_EXPORT int
GroupedArrayFloat32_RollingStdTransform(GroupedArrayHandle handle, int lag,
                                        int window_size, int min_samples,
                                        float *out);
DLL_EXPORT int
GroupedArrayFloat64_RollingStdTransform(GroupedArrayHandle handle, int lag,
                                        int window_size, int min_samples,
                                        double *out);

DLL_EXPORT int
GroupedArrayFloat32_RollingMinTransform(GroupedArrayHandle handle, int lag,
                                        int window_size, int min_samples,
                                        float *out);
DLL_EXPORT int
GroupedArrayFloat64_RollingMinTransform(GroupedArrayHandle handle, int lag,
                                        int window_size, int min_samples,
                                        double *out);

DLL_EXPORT int
GroupedArrayFloat32_RollingMaxTransform(GroupedArrayHandle handle, int lag,
                                        int window_size, int min_samples,
                                        float *out);
DLL_EXPORT int
GroupedArrayFloat64_RollingMaxTransform(GroupedArrayHandle handle, int lag,
                                        int window_size, int min_samples,
                                        double *out);

DLL_EXPORT int
GroupedArrayFloat32_RollingQuantileTransform(GroupedArrayHandle handle, int lag,
                                             float p, int window_size,
                                             int min_samples, float *out);
DLL_EXPORT int
GroupedArrayFloat64_RollingQuantileTransform(GroupedArrayHandle handle, int lag,
                                             double p, int window_size,
                                             int min_samples, double *out);

DLL_EXPORT int GroupedArrayFloat32_RollingMeanUpdate(GroupedArrayHandle handle,
                                                     int lag, int window_size,
                                                     int min_samples,
                                                     float *out);
DLL_EXPORT int GroupedArrayFloat64_RollingMeanUpdate(GroupedArrayHandle handle,
                                                     int lag, int window_size,
                                                     int min_samples,
                                                     double *out);

DLL_EXPORT int GroupedArrayFloat32_RollingStdUpdate(GroupedArrayHandle handle,
                                                    int lag, int window_size,
                                                    int min_samples,
                                                    float *out);
DLL_EXPORT int GroupedArrayFloat64_RollingStdUpdate(GroupedArrayHandle handle,
                                                    int lag, int window_size,
                                                    int min_samples,
                                                    double *out);

DLL_EXPORT int GroupedArrayFloat32_RollingMinUpdate(GroupedArrayHandle handle,
                                                    int lag, int window_size,
                                                    int min_samples,
                                                    float *out);
DLL_EXPORT int GroupedArrayFloat64_RollingMinUpdate(GroupedArrayHandle handle,
                                                    int lag, int window_size,
                                                    int min_samples,
                                                    double *out);

DLL_EXPORT int GroupedArrayFloat32_RollingMaxUpdate(GroupedArrayHandle handle,
                                                    int lag, int window_size,
                                                    int min_samples,
                                                    float *out);
DLL_EXPORT int GroupedArrayFloat64_RollingMaxUpdate(GroupedArrayHandle handle,
                                                    int lag, int window_size,
                                                    int min_samples,
                                                    double *out);

DLL_EXPORT int
GroupedArrayFloat32_RollingQuantileUpdate(GroupedArrayHandle handle, int lag,
                                          float p, int window_size,
                                          int min_samples, float *out);
DLL_EXPORT int
GroupedArrayFloat64_RollingQuantileUpdate(GroupedArrayHandle handle, int lag,
                                          double p, int window_size,
                                          int min_samples, double *out);

DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingMeanTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingMeanTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, double *out);

DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingStdTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingStdTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, double *out);

DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingMinTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingMinTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, double *out);

DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingMaxTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingMaxTransform(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, double *out);

DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingQuantileTransform(
    GroupedArrayHandle handle, int lag, int season_length, float p,
    int window_size, int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingQuantileTransform(
    GroupedArrayHandle handle, int lag, int season_length, double p,
    int window_size, int min_samples, double *out);

DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingMeanUpdate(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingMeanUpdate(
    GroupedArrayHandle handle, int lag, int season_length, int window_size,
    int min_samples, double *out);

DLL_EXPORT int
GroupedArrayFloat32_SeasonalRollingStdUpdate(GroupedArrayHandle handle, int lag,
                                             int season_length, int window_size,
                                             int min_samples, float *out);
DLL_EXPORT int
GroupedArrayFloat64_SeasonalRollingStdUpdate(GroupedArrayHandle handle, int lag,
                                             int season_length, int window_size,
                                             int min_samples, double *out);

DLL_EXPORT int
GroupedArrayFloat32_SeasonalRollingMinUpdate(GroupedArrayHandle handle, int lag,
                                             int season_length, int window_size,
                                             int min_samples, float *out);
DLL_EXPORT int
GroupedArrayFloat64_SeasonalRollingMinUpdate(GroupedArrayHandle handle, int lag,
                                             int season_length, int window_size,
                                             int min_samples, double *out);

DLL_EXPORT int
GroupedArrayFloat32_SeasonalRollingMaxUpdate(GroupedArrayHandle handle, int lag,
                                             int season_length, int window_size,
                                             int min_samples, float *out);
DLL_EXPORT int
GroupedArrayFloat64_SeasonalRollingMaxUpdate(GroupedArrayHandle handle, int lag,
                                             int season_length, int window_size,
                                             int min_samples, double *out);

DLL_EXPORT int GroupedArrayFloat32_SeasonalRollingQuantileUpdate(
    GroupedArrayHandle handle, int lag, int season_length, float p,
    int window_size, int min_samples, float *out);
DLL_EXPORT int GroupedArrayFloat64_SeasonalRollingQuantileUpdate(
    GroupedArrayHandle handle, int lag, int season_length, double p,
    int window_size, int min_samples, double *out);

DLL_EXPORT int
GroupedArrayFloat32_ExpandingMeanTransform(GroupedArrayHandle handle, int lag,
                                           float *out, float *agg);
DLL_EXPORT int
GroupedArrayFloat64_ExpandingMeanTransform(GroupedArrayHandle handle, int lag,
                                           double *out, double *agg);

DLL_EXPORT int
GroupedArrayFloat32_ExpandingStdTransform(GroupedArrayHandle handle, int lag,
                                          float *out, float *agg);
DLL_EXPORT int
GroupedArrayFloat64_ExpandingStdTransform(GroupedArrayHandle handle, int lag,
                                          double *out, double *agg);

DLL_EXPORT int
GroupedArrayFloat32_ExpandingMinTransform(GroupedArrayHandle handle, int lag,
                                          float *out);
DLL_EXPORT int
GroupedArrayFloat64_ExpandingMinTransform(GroupedArrayHandle handle, int lag,
                                          double *out);

DLL_EXPORT int
GroupedArrayFloat32_ExpandingMaxTransform(GroupedArrayHandle handle, int lag,
                                          float *out);
DLL_EXPORT int
GroupedArrayFloat64_ExpandingMaxTransform(GroupedArrayHandle handle, int lag,
                                          double *out);

DLL_EXPORT int
GroupedArrayFloat32_ExpandingQuantileTransform(GroupedArrayHandle handle,
                                               int lag, float p, float *out);
DLL_EXPORT int
GroupedArrayFloat64_ExpandingQuantileTransform(GroupedArrayHandle handle,
                                               int lag, double p, double *out);

DLL_EXPORT int
GroupedArrayFloat32_ExpandingQuantileUpdate(GroupedArrayHandle handle, int lag,
                                            float p, float *out);
DLL_EXPORT int
GroupedArrayFloat64_ExpandingQuantileUpdate(GroupedArrayHandle handle, int lag,
                                            double p, double *out);

DLL_EXPORT int GroupedArrayFloat32_ExponentiallyWeightedMeanTransform(
    GroupedArrayHandle handle, int lag, float alpha, float *out);
DLL_EXPORT int GroupedArrayFloat64_ExponentiallyWeightedMeanTransform(
    GroupedArrayHandle handle, int lag, double alpha, double *out);
}
