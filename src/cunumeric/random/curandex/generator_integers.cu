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
struct integers;

template <>
struct integers<int32_t> {
  int32_t from;
  int32_t to;

  template <typename gen_t>
  __forceinline__ __host__ __device__ int32_t operator()(gen_t& gen)
  {
    // take two draws to get a 64 bits value
    return (int32_t)(curand(&gen) % (uint32_t)(to - from)) + from;
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateIntegers32Ex(
  curandGeneratorEx_t generator, int32_t* outputPtr, size_t n, int32_t low, int32_t high)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  integers<int32_t> func;
  func.from = low;
  func.to   = high;
  return curandimpl::dispatch_sample<integers<int32_t>, int32_t>(gen, func, n, outputPtr);
}

template <>
struct integers<int64_t> {
  int64_t from;
  int64_t to;

  template <typename gen_t>
  __forceinline__ __host__ __device__ int64_t operator()(gen_t& gen)
  {
    // take two draws to get a 64 bits value
    unsigned low  = curand(&gen);
    unsigned high = curand(&gen);
    return (int64_t)((((uint64_t)high << 32) | (uint64_t)low) % (to - from)) + from;
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateIntegers64Ex(
  curandGeneratorEx_t generator, int64_t* outputPtr, size_t n, int64_t low, int64_t high)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  integers<int64_t> func;
  func.from = low;
  func.to   = high;
  return curandimpl::dispatch_sample<integers<int64_t>, int64_t>(gen, func, n, outputPtr);
}
