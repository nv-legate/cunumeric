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
extern "C" curandStatus_t CURANDAPI randutilCreateGenerator(randutilGenerator_t* generator,
                                                            curandRngType_t rng_type,
                                                            uint64_t seed,
                                                            uint64_t generatorID,
                                                            uint32_t flags,
                                                            cudaStream_t stream);
#endif

extern "C" curandStatus_t CURANDAPI randutilCreateGeneratorHost(randutilGenerator_t* generator,
                                                                curandRngType_t rng_type,
                                                                uint64_t seed,
                                                                uint64_t generatorID,
                                                                uint32_t flags);
extern "C" curandStatus_t CURANDAPI randutilDestroyGenerator(randutilGenerator_t generator);

/* curand distributions */

extern "C" curandStatus_t CURANDAPI randutilGenerateIntegers32(randutilGenerator_t generator,
                                                               int32_t* outputPtr,
                                                               size_t num,
                                                               int32_t low /* inclusive */,
                                                               int32_t high /* exclusive */);
extern "C" curandStatus_t CURANDAPI randutilGenerateIntegers64(randutilGenerator_t generator,
                                                               int64_t* outputPtr,
                                                               size_t num,
                                                               int64_t low /* inclusive */,
                                                               int64_t high /* exclusive */);
extern "C" curandStatus_t CURANDAPI randutilGenerateRawUInt32(randutilGenerator_t generator,
                                                              uint32_t* outputPtr,
                                                              size_t num);

extern "C" curandStatus_t CURANDAPI randutilGenerateUniformEx(randutilGenerator_t generator,
                                                              float* outputPtr,
                                                              size_t num,
                                                              float low  = 0.0f, /* inclusive */
                                                              float high = 1.0f /* exclusive */);

extern "C" curandStatus_t CURANDAPI
randutilGenerateUniformDoubleEx(randutilGenerator_t generator,
                                double* outputPtr,
                                size_t num,
                                double low  = 0.0, /* inclusive */
                                double high = 1.0 /* exclusive */);

extern "C" curandStatus_t CURANDAPI randutilGenerateLogNormalEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev);
extern "C" curandStatus_t CURANDAPI randutilGenerateLogNormalDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev);

extern "C" curandStatus_t CURANDAPI randutilGenerateNormalEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev);
extern "C" curandStatus_t CURANDAPI randutilGenerateNormalDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev);

extern "C" curandStatus_t CURANDAPI randutilGeneratePoissonEx(randutilGenerator_t generator,
                                                              uint32_t* outputPtr,
                                                              size_t n,
                                                              double lambda);

/* Straightforward Distributions */

extern "C" curandStatus_t CURANDAPI randutilGenerateExponentialEx(randutilGenerator_t generator,
                                                                  float* outputPtr,
                                                                  size_t n,
                                                                  float scale);
extern "C" curandStatus_t CURANDAPI randutilGenerateExponentialDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double scale);

extern "C" curandStatus_t CURANDAPI randutilGenerateGumbelEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float beta);
extern "C" curandStatus_t CURANDAPI randutilGenerateGumbelDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double beta);

extern "C" curandStatus_t CURANDAPI randutilGenerateLaplaceEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float beta);
extern "C" curandStatus_t CURANDAPI randutilGenerateLaplaceDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double beta);
