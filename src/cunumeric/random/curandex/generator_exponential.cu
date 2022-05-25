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
struct exponential_t;

template <>
struct exponential_t<float> {
  float scale = 1.0f;

  template <typename gen_t>
  __forceinline__ __host__ __device__ float operator()(gen_t& gen)
  {
    float uni = curand_uniform(&gen);
    return -::logf(uni) * scale;
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateExponentialEx(curandGeneratorEx_t generator,
                                                                float* outputPtr,
                                                                size_t n,
                                                                float scale)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  exponential_t<float> func;
  func.scale = scale;
  return curandimpl::dispatch_sample<exponential_t<float>, float>(gen, func, n, outputPtr);
}

template <>
struct exponential_t<double> {
  double scale = 1.0f;

  template <typename gen_t>
  __forceinline__ __host__ __device__ double operator()(gen_t& gen)
  {
    double uni = curand_uniform_double(&gen);
    return -::logf(uni) * scale;
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateExponentialDoubleEx(curandGeneratorEx_t generator,
                                                                      double* outputPtr,
                                                                      size_t n,
                                                                      double scale)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  exponential_t<double> func;
  func.scale = scale;
  return curandimpl::dispatch_sample<exponential_t<double>, double>(gen, func, n, outputPtr);
}
