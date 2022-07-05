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

#include "generator.h"

template <typename field_t>
struct cauchy_t;

template <>
struct cauchy_t<float> {
  static constexpr float pi = 3.1415926535897932384626433832795f;

  float x0, gamma;

  template <typename gen_t>
  __forceinline__ __host__ __device__ float operator()(gen_t& gen)
  {
    float y = curand_uniform(&gen);  // y cannot be 0
    return x0 + gamma * ::tanf(pi * (y - 0.5f));
  }
};

template <>
struct cauchy_t<double> {
  static constexpr double pi = 3.1415926535897932384626433832795;

  double x0, gamma;

  template <typename gen_t>
  __forceinline__ __host__ __device__ double operator()(gen_t& gen)
  {
    double y = curand_uniform_double(&gen);  // y cannot be 0
    return x0 + gamma * ::tan(pi * (y - 0.5));
  }
};
