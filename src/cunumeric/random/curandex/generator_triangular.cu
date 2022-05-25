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
struct triangular_t;

template <>
struct triangular_t<float> {
  float a, b, c;

  template <typename gen_t>
  __forceinline__ __host__ __device__ float operator()(gen_t& gen)
  {
    float y = curand_uniform(&gen);  // y cannot be 0
    if (y <= ((c - a) / (b - a))) {
      float delta = (y * (b - a) * (c - a));
      if (delta < 0.0f) delta = 0.0f;
      return a + ::sqrtf(delta);
    } else {
      float delta = ((1.0f - y) * (b - a) * (b - c));
      if (delta < 0.0f) delta = 0.0f;
      return b - ::sqrtf(delta);
    }
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateTriangularEx(
  curandGeneratorEx_t generator, float* outputPtr, size_t num, float a, float c, float b)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  triangular_t<float> func;
  func.a = a;
  func.b = b;
  func.c = c;
  return curandimpl::dispatch_sample<triangular_t<float>, float>(gen, func, num, outputPtr);
}

template <>
struct triangular_t<double> {
  double a, b, c;

  template <typename gen_t>
  __forceinline__ __host__ __device__ double operator()(gen_t& gen)
  {
    double y = curand_uniform_double(&gen);  // y cannot be 0
    if (y <= ((c - a) / (b - a))) {
      double delta = (y * (b - a) * (c - a));
      if (delta < 0.0) delta = 0.0;
      return a + ::sqrt(delta);
    } else {
      double delta = ((1.0 - y) * (b - a) * (b - c));
      if (delta < 0.0) delta = 0.0;
      return b - ::sqrt(delta);
    }
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateTriangularDoubleEx(
  curandGeneratorEx_t generator, double* outputPtr, size_t num, double a, double c, double b)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  triangular_t<double> func;
  func.a = a;
  func.b = b;
  func.c = c;
  return curandimpl::dispatch_sample<triangular_t<double>, double>(gen, func, num, outputPtr);
}
