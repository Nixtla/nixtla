#pragma once

#include "export.h"
#include "grouped_array.h"

extern "C" {
DLL_EXPORT float Float32_BoxCoxLambdaGuerrero(const float *x, int n, int period,
                                              float lower, float upper);
DLL_EXPORT double Float64_BoxCoxLambdaGuerrero(const double *x, int n,
                                               int period, double lower,
                                               double upper);

DLL_EXPORT float Float32_BoxCoxLambdaLogLik(const float *x, int n, float lower,
                                            float upper);
DLL_EXPORT double Float64_BoxCoxLambdaLogLik(const double *x, int n,
                                             double lower, double upper);

DLL_EXPORT void Float32_BoxCoxTransform(const float *x, int n, float lambda,
                                        float *out);
DLL_EXPORT void Float64_BoxCoxTransform(const double *x, int n, double lambda,
                                        double *out);

DLL_EXPORT void Float32_BoxCoxInverseTransform(const float *x, int n,
                                               float lambda, float *out);
DLL_EXPORT void Float64_BoxCoxInverseTransform(const double *x, int n,
                                               double lambda, double *out);

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

DLL_EXPORT int
GroupedArrayFloat32_BoxCoxLambdaGuerrero(GroupedArrayHandle handle, int period,
                                         float lower, float upper, float *out);

DLL_EXPORT int
GroupedArrayFloat64_BoxCoxLambdaGuerrero(GroupedArrayHandle handle, int period,
                                         double lower, double upper,
                                         double *out);

DLL_EXPORT void
GroupedArrayFloat32_BoxCoxLambdaLogLik(GroupedArrayHandle handle, float lower,
                                       float upper, float *out);
DLL_EXPORT void
GroupedArrayFloat64_BoxCoxLambdaLogLik(GroupedArrayHandle handle, double lower,
                                       double upper, double *out);

DLL_EXPORT int GroupedArrayFloat32_BoxCoxTransform(GroupedArrayHandle handle,
                                                   const float *lambdas,
                                                   float *out);
DLL_EXPORT int GroupedArrayFloat64_BoxCoxTransform(GroupedArrayHandle handle,
                                                   const double *lambdas,
                                                   double *out);

DLL_EXPORT int
GroupedArrayFloat32_BoxCoxInverseTransform(GroupedArrayHandle handle,
                                           const float *lambdas, float *out);
DLL_EXPORT int
GroupedArrayFloat64_BoxCoxInverseTransform(GroupedArrayHandle handle,
                                           const double *lambdas, double *out);
}
