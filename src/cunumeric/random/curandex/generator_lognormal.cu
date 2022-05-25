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

#include "generator.cuh"

template <typename field_t>
struct lognormal_t;

template <>
struct lognormal_t<float> {
  float mean   = 0.0;
  float stddev = 1.0;

  template <typename gen_t>
  __forceinline__ __host__ __device__ float operator()(gen_t& gen)
  {
    return curand_log_normal(&gen, mean, stddev);
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateLogNormalEx(
  curandGeneratorEx_t generator, float* outputPtr, size_t n, float mean, float stddev)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  lognormal_t<float> func;
  func.mean   = mean;
  func.stddev = stddev;
  return curandimpl::dispatch_sample<lognormal_t<float>, float>(gen, func, n, outputPtr);
}

template <>
struct lognormal_t<double> {
  double mean   = 0.0;
  double stddev = 1.0;

  template <typename gen_t>
  __forceinline__ __host__ __device__ double operator()(gen_t& gen)
  {
    return curand_log_normal_double(&gen, mean, stddev);
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateLogNormalDoubleEx(
  curandGeneratorEx_t generator, double* outputPtr, size_t n, double mean, double stddev)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  lognormal_t<double> func;
  func.mean   = mean;
  func.stddev = stddev;
  return curandimpl::dispatch_sample<lognormal_t<double>, double>(gen, func, n, outputPtr);
}