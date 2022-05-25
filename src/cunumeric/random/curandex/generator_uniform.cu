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
struct uniform_t;

template <>
struct uniform_t<float> {
  float offset, mult;

  template <typename gen_t>
  __forceinline__ __host__ __device__ float operator()(gen_t& gen)
  {
    return offset + mult * curand_uniform(&gen);
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateUniformEx(
  curandGeneratorEx_t generator, float* outputPtr, size_t n, float low, float high)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  uniform_t<float> func;
  func.offset = low;
  func.mult   = high - low;
  return curandimpl::dispatch_sample<uniform_t<float>, float>(gen, func, n, outputPtr);
}

template <>
struct uniform_t<double> {
  double offset, mult;

  template <typename gen_t>
  __forceinline__ __host__ __device__ double operator()(gen_t& gen)
  {
    return offset + mult * curand_uniform_double(&gen);
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateUniformDoubleEx(
  curandGeneratorEx_t generator, double* outputPtr, size_t n, double low, double high)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  uniform_t<double> func;
  func.offset = low;
  func.mult   = high - low;
  return curandimpl::dispatch_sample<uniform_t<double>, double>(gen, func, n, outputPtr);
}