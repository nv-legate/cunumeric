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

typedef void* curandGeneratorEx_t;

/* generator */

extern "C" curandStatus_t CURANDAPI curandCreateGeneratorEx(curandGeneratorEx_t* generator,
                                                            curandRngType_t rng_type,
                                                            uint64_t seed,
                                                            uint64_t generatorID,
                                                            uint32_t flags,
                                                            cudaStream_t stream);
extern "C" curandStatus_t CURANDAPI curandCreateGeneratorHostEx(curandGeneratorEx_t* generator,
                                                                curandRngType_t rng_type,
                                                                uint64_t seed,
                                                                uint64_t generatorID,
                                                                uint32_t flags);
extern "C" curandStatus_t CURANDAPI curandDestroyGeneratorEx(curandGeneratorEx_t generator);

/* curand distributions */

extern "C" curandStatus_t CURANDAPI curandGenerateNormalEx(
  curandGeneratorEx_t generator, float* outputPtr, size_t n, float mean, float stddev);
extern "C" curandStatus_t CURANDAPI curandGenerateNormalDoubleEx(
  curandGeneratorEx_t generator, double* outputPtr, size_t n, double mean, double stddev);

extern "C" curandStatus_t CURANDAPI curandGenerateLogNormalEx(
  curandGeneratorEx_t generator, float* outputPtr, size_t n, float mean, float stddev);
extern "C" curandStatus_t CURANDAPI curandGenerateLogNormalDoubleEx(
  curandGeneratorEx_t generator, double* outputPtr, size_t n, double mean, double stddev);

extern "C" curandStatus_t CURANDAPI curandGeneratePoissonEx(curandGeneratorEx_t generator,
                                                            uint32_t* outputPtr,
                                                            size_t n,
                                                            double lambda);

extern "C" curandStatus_t CURANDAPI
curandGenerateUniformEx(curandGeneratorEx_t generator,
                        float* outputPtr,
                        size_t num,
                        float low  = 0.0f,
                        float high = 1.0f); /* low exclusive, high inclusive */
extern "C" curandStatus_t CURANDAPI
curandGenerateUniformDoubleEx(curandGeneratorEx_t generator,
                              double* outputPtr,
                              size_t num,
                              double low  = 0.0,
                              double high = 1.0); /* low exclusive, high inclusive */

extern "C" curandStatus_t CURANDAPI curandGenerateLongLongEx(curandGeneratorEx_t generator,
                                                             uint64_t* outputPtr,
                                                             size_t num);

extern "C" curandStatus_t CURANDAPI curandGenerateIntegers32Ex(curandGeneratorEx_t generator,
                                                               int32_t* outputPtr,
                                                               size_t num,
                                                               int32_t low /* inclusive */,
                                                               int32_t high /* exclusive */);
extern "C" curandStatus_t CURANDAPI curandGenerateIntegers64Ex(curandGeneratorEx_t generator,
                                                               int64_t* outputPtr,
                                                               size_t num,
                                                               int64_t low /* inclusive */,
                                                               int64_t high /* exclusive */);

extern "C" curandStatus_t CURANDAPI curandGenerateRawUInt32Ex(curandGeneratorEx_t generator,
                                                              uint32_t* outputPtr,
                                                              size_t num);

/* straightforward distributions  */

extern "C" curandStatus_t CURANDAPI curandGenerateExponentialEx(curandGeneratorEx_t generator,
                                                                float* outputPtr,
                                                                size_t n,
                                                                float scale);
extern "C" curandStatus_t CURANDAPI curandGenerateExponentialDoubleEx(curandGeneratorEx_t generator,
                                                                      double* outputPtr,
                                                                      size_t n,
                                                                      double scale);

extern "C" curandStatus_t CURANDAPI curandGenerateGumbelEx(
  curandGeneratorEx_t generator, float* outputPtr, size_t num, float mu = 0.0f, float beta = 1.0f);
extern "C" curandStatus_t CURANDAPI curandGenerateGumbelDoubleEx(
  curandGeneratorEx_t generator, double* outputPtr, size_t num, double mu = 0.0, double beta = 1.0);

extern "C" curandStatus_t CURANDAPI curandGenerateLaplaceEx(
  curandGeneratorEx_t generator, float* outputPtr, size_t num, float mu = 0.0f, float beta = 1.0f);
extern "C" curandStatus_t CURANDAPI curandGenerateLaplaceDoubleEx(
  curandGeneratorEx_t generator, double* outputPtr, size_t num, double mu = 0.0, double beta = 1.0);

extern "C" curandStatus_t CURANDAPI curandGenerateLogisticEx(
  curandGeneratorEx_t generator, float* outputPtr, size_t num, float mu = 0.0f, float beta = 1.0f);
extern "C" curandStatus_t CURANDAPI curandGenerateLogisticDoubleEx(
  curandGeneratorEx_t generator, double* outputPtr, size_t num, double mu = 0.0, double beta = 1.0);

extern "C" curandStatus_t CURANDAPI curandGenerateParetoEx(
  curandGeneratorEx_t generator, float* outputPtr, size_t num, float xm, float alpha);
extern "C" curandStatus_t CURANDAPI curandGenerateParetoDoubleEx(
  curandGeneratorEx_t generator, double* outputPtr, size_t num, double xm, double alpha);

extern "C" curandStatus_t CURANDAPI curandGeneratePowerEx(curandGeneratorEx_t generator,
                                                          float* outputPtr,
                                                          size_t num,
                                                          float alpha);
extern "C" curandStatus_t CURANDAPI curandGeneratePowerDoubleEx(curandGeneratorEx_t generator,
                                                                double* outputPtr,
                                                                size_t num,
                                                                double alpha);

extern "C" curandStatus_t CURANDAPI curandGenerateRayleighEx(curandGeneratorEx_t generator,
                                                             float* outputPtr,
                                                             size_t num,
                                                             float sigma);
extern "C" curandStatus_t CURANDAPI curandGenerateRayleighDoubleEx(curandGeneratorEx_t generator,
                                                                   double* outputPtr,
                                                                   size_t num,
                                                                   double sigma);

extern "C" curandStatus_t CURANDAPI curandGenerateCauchyEx(
  curandGeneratorEx_t generator, float* outputPtr, size_t num, float x0 = 0.0f, float gamma = 1.0f);
extern "C" curandStatus_t CURANDAPI curandGenerateCauchyDoubleEx(curandGeneratorEx_t generator,
                                                                 double* outputPtr,
                                                                 size_t num,
                                                                 double x0    = 0.0,
                                                                 double gamma = 1.0);

extern "C" curandStatus_t CURANDAPI curandGenerateTriangularEx(curandGeneratorEx_t generator,
                                                               float* outputPtr,
                                                               size_t num,
                                                               float a,
                                                               float c,
                                                               float b);  // a <= c <= b, a < b
extern "C" curandStatus_t CURANDAPI curandGenerateTriangularDoubleEx(
  curandGeneratorEx_t generator, double* outputPtr, size_t num, double a, double c, double b);

extern "C" curandStatus_t CURANDAPI curandGenerateWeibullEx(
  curandGeneratorEx_t generator, float* outputPtr, size_t num, float lambda, float k);
extern "C" curandStatus_t CURANDAPI curandGenerateWeibullDoubleEx(
  curandGeneratorEx_t generator, double* outputPtr, size_t num, double lambda, double k);
