/* Copyright 2021-2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#pragma once

#include <cstdint>
#include <curand.h>

typedef void* randutilGenerator_t;

/* generator */

// CUDA-ONLY API
#ifdef LEGATE_USE_CUDA
extern "C" curandStatus_t randutilCreateGenerator(randutilGenerator_t* generator,
                                                  curandRngType_t rng_type,
                                                  uint64_t seed,
                                                  uint64_t generatorID,
                                                  uint32_t flags,
                                                  cudaStream_t stream);
#endif

extern "C" curandStatus_t randutilCreateGeneratorHost(randutilGenerator_t* generator,
                                                      curandRngType_t rng_type,
                                                      uint64_t seed,
                                                      uint64_t generatorID,
                                                      uint32_t flags);
extern "C" curandStatus_t randutilDestroyGenerator(randutilGenerator_t generator);

/* curand distributions */

extern "C" curandStatus_t randutilGenerateIntegers16(randutilGenerator_t generator,
                                                     int16_t* outputPtr,
                                                     size_t num,
                                                     int16_t low /* inclusive */,
                                                     int16_t high /* exclusive */);

extern "C" curandStatus_t randutilGenerateIntegers32(randutilGenerator_t generator,
                                                     int32_t* outputPtr,
                                                     size_t num,
                                                     int32_t low /* inclusive */,
                                                     int32_t high /* exclusive */);
extern "C" curandStatus_t randutilGenerateIntegers64(randutilGenerator_t generator,
                                                     int64_t* outputPtr,
                                                     size_t num,
                                                     int64_t low /* inclusive */,
                                                     int64_t high /* exclusive */);
extern "C" curandStatus_t randutilGenerateRawUInt32(randutilGenerator_t generator,
                                                    uint32_t* outputPtr,
                                                    size_t num);

extern "C" curandStatus_t randutilGenerateUniformEx(randutilGenerator_t generator,
                                                    float* outputPtr,
                                                    size_t num,
                                                    float low  = 0.0f, /* inclusive */
                                                    float high = 1.0f /* exclusive */);

extern "C" curandStatus_t randutilGenerateUniformDoubleEx(randutilGenerator_t generator,
                                                          double* outputPtr,
                                                          size_t num,
                                                          double low  = 0.0, /* inclusive */
                                                          double high = 1.0 /* exclusive */);

extern "C" curandStatus_t randutilGenerateLogNormalEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev);
extern "C" curandStatus_t randutilGenerateLogNormalDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev);

extern "C" curandStatus_t randutilGenerateNormalEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev);
extern "C" curandStatus_t randutilGenerateNormalDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev);

extern "C" curandStatus_t randutilGeneratePoissonEx(randutilGenerator_t generator,
                                                    uint32_t* outputPtr,
                                                    size_t n,
                                                    double lambda);

/* Straightforward Distributions */

extern "C" curandStatus_t randutilGenerateExponentialEx(randutilGenerator_t generator,
                                                        float* outputPtr,
                                                        size_t n,
                                                        float scale);
extern "C" curandStatus_t randutilGenerateExponentialDoubleEx(randutilGenerator_t generator,
                                                              double* outputPtr,
                                                              size_t n,
                                                              double scale);

extern "C" curandStatus_t randutilGenerateGumbelEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float beta);
extern "C" curandStatus_t randutilGenerateGumbelDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double beta);

extern "C" curandStatus_t randutilGenerateLaplaceEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float beta);
extern "C" curandStatus_t randutilGenerateLaplaceDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double beta);

extern "C" curandStatus_t randutilGenerateLogisticEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float beta);
extern "C" curandStatus_t randutilGenerateLogisticDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double beta);

extern "C" curandStatus_t randutilGenerateParetoEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float xm, float alpha);
extern "C" curandStatus_t randutilGenerateParetoDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double xm, double alpha);

extern "C" curandStatus_t randutilGeneratePowerEx(randutilGenerator_t generator,
                                                  float* outputPtr,
                                                  size_t n,
                                                  float alpha);
extern "C" curandStatus_t randutilGeneratePowerDoubleEx(randutilGenerator_t generator,
                                                        double* outputPtr,
                                                        size_t n,
                                                        double alpha);

extern "C" curandStatus_t randutilGenerateRayleighEx(randutilGenerator_t generator,
                                                     float* outputPtr,
                                                     size_t n,
                                                     float sigma);
extern "C" curandStatus_t randutilGenerateRayleighDoubleEx(randutilGenerator_t generator,
                                                           double* outputPtr,
                                                           size_t n,
                                                           double sigma);

extern "C" curandStatus_t randutilGenerateCauchyEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float x0, float gamma);
extern "C" curandStatus_t randutilGenerateCauchyDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double x0, double gamma);

extern "C" curandStatus_t randutilGenerateTriangularEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float a, float b, float c);
extern "C" curandStatus_t randutilGenerateTriangularDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double a, double b, double c);

extern "C" curandStatus_t randutilGenerateWeibullEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float lam, float k);
extern "C" curandStatus_t randutilGenerateWeibullDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double lam, double k);

/* more advanced distributions */

extern "C" curandStatus_t randutilGenerateBetaEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float a, float b);
extern "C" curandStatus_t randutilGenerateBetaDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double a, double b);

extern "C" curandStatus_t randutilGenerateFisherSnedecorEx(
  randutilGenerator_t generator,
  float* outputPtr,
  size_t n,
  float dfnum,
  float dfden,
  float nonc = 0.0f);  // 0.0f is F distribution
extern "C" curandStatus_t randutilGenerateFisherSnedecorDoubleEx(
  randutilGenerator_t generator,
  double* outputPtr,
  size_t n,
  double dfnum,
  double dfden,
  double nonc = 0.0);  // 0.0 is F distribution

extern "C" curandStatus_t randutilGenerateLogSeriesEx(randutilGenerator_t generator,
                                                      uint32_t* outputPtr,
                                                      size_t n,
                                                      double p);

extern "C" curandStatus_t randutilGenerateChiSquareEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float df, float nonc = 0.0);
extern "C" curandStatus_t randutilGenerateChiSquareDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double df, double nonc = 0.0);

extern "C" curandStatus_t randutilGenerateGammaEx(
  randutilGenerator_t generator,
  float* outputPtr,
  size_t n,
  float shape,
  float scale = 1.0f);  // scale = 1.0 is standard_gamma
extern "C" curandStatus_t randutilGenerateGammaDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double shape, double scale = 1.0);

extern "C" curandStatus_t randutilGenerateStandardTDoubleEx(randutilGenerator_t generator,
                                                            double* outputPtr,
                                                            size_t n,
                                                            double df);
extern "C" curandStatus_t randutilGenerateStandardTEx(randutilGenerator_t generator,
                                                      float* outputPtr,
                                                      size_t n,
                                                      float df);
extern "C" curandStatus_t randutilGenerateHyperGeometricEx(randutilGenerator_t generator,
                                                           uint32_t* outputPtr,
                                                           size_t n,
                                                           int64_t ngood,
                                                           int64_t nbad,
                                                           int64_t nsample);
extern "C" curandStatus_t randutilGenerateVonMisesDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double kappa);
extern "C" curandStatus_t randutilGenerateVonMisesEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float kappa);
extern "C" curandStatus_t randutilGenerateZipfEx(randutilGenerator_t generator,
                                                 uint32_t* outputPtr,
                                                 size_t n,
                                                 double a);
extern "C" curandStatus_t randutilGenerateGeometricEx(randutilGenerator_t generator,
                                                      uint32_t* outputPtr,
                                                      size_t n,
                                                      double p);
extern "C" curandStatus_t randutilGenerateWaldDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double lambda);
extern "C" curandStatus_t randutilGenerateWaldEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float lambda);

extern "C" curandStatus_t randutilGenerateBinomialEx(
  randutilGenerator_t generator, uint32_t* outputPtr, size_t n, uint32_t ntrials, double p);
extern "C" curandStatus_t randutilGenerateNegativeBinomialEx(
  randutilGenerator_t generator, uint32_t* outputPtr, size_t n, uint32_t ntrials, double p);
