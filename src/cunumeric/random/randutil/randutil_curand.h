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

// This header file imports some head-only part of curand for the HOST-side implementation of
// generators

// also allow usage of generators on host
#ifdef LEGATE_USE_CUDA

#define QUALIFIERS static __forceinline__ __device__ __host__
#define RANDUTIL_QUALIFIERS __forceinline__ __device__ __host__
#include <curand_kernel.h>

#else
// host generators are not compiled with nvcc
#define QUALIFIERS static
#define RANDUTIL_QUALIFIERS
#pragma region avoid inclusion of cuda_runtime.h in curand.h, and mtgp32

// include these before defining CUDACC_RTC to still enable host-side precalc matrix
#define __device__
#include <curand_precalc.h>
#include <curand_mrg32k3a.h>
#undef __device__

#define __CUDACC_RTC__
#include <cstring>
#include <cmath>
#pragma endregion

// curand_mtgp32_kernel.h
struct dim3 {
  int x, y, z;
};
struct uint3 {
  unsigned int x, y, z;
};

// curand_philox4x32_x.h
struct alignas(16) uint4 {
  unsigned int x, y, z, w;
};
struct alignas(8) uint2 {
  unsigned int x, y;
};
#define make_uint4(a, b, c, d) uint4({a, b, c, d})
// curand_uniform.h
struct alignas(16) float4 {
  float x, y, z, w;
};
struct alignas(16) double2 {
  double x, y;
};
struct alignas(32) double4 {
  double x, y, z, w;
};

// curand_normal_static.h
#define __device__
#define __host__
#define __forceinline__
#include <curand_normal_static.h>
#undef __device__
#undef __host__
#undef __forceinline__

// curand_normal.h
struct alignas(8) float2 {
  float x, y;
};

// curand_poisson.h
struct alignas(16) int4 {
  int x, y, z, w;
};

#define __device__
#define __constant__  // used in curand_poisson.h, cannot be pre-included because of other
                      // dependencies
#include <curand_kernel.h>
#undef __constant__
#undef __device__

typedef void* cudaStream_t;

#endif