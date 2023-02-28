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

#include "legate.h"
#include "core/cuda/cuda_help.h"
#include "core/cuda/stream_pool.h"
#include "cunumeric/arg.h"
#include "cunumeric/arg.inl"
#include "cunumeric/device_scalar_reduction_buffer.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cutensor.h>
#include <nccl.h>

#define THREADS_PER_BLOCK 128
#define MIN_CTAS_PER_SM 4
#define MAX_REDUCTION_CTAS 1024
#define COOPERATIVE_THREADS 256
#define COOPERATIVE_CTAS_PER_SM 4

#define CHECK_CUBLAS(expr)                        \
  do {                                            \
    cublasStatus_t __result__ = (expr);           \
    check_cublas(__result__, __FILE__, __LINE__); \
  } while (false)

#define CHECK_CUFFT(expr)                        \
  do {                                           \
    cufftResult __result__ = (expr);             \
    check_cufft(__result__, __FILE__, __LINE__); \
  } while (false)

#define CHECK_CUSOLVER(expr)                        \
  do {                                              \
    cusolverStatus_t __result__ = (expr);           \
    check_cusolver(__result__, __FILE__, __LINE__); \
  } while (false)

#define CHECK_CUTENSOR(expr)                        \
  do {                                              \
    cutensorStatus_t __result__ = (expr);           \
    check_cutensor(__result__, __FILE__, __LINE__); \
  } while (false)

#define CHECK_NCCL(expr)                    \
  do {                                      \
    ncclResult_t result = (expr);           \
    check_nccl(result, __FILE__, __LINE__); \
  } while (false)

#ifndef MAX
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif
#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

namespace cunumeric {

__device__ inline size_t global_tid_1d()
{
  return static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

struct cufftPlan {
  cufftHandle handle;
  size_t workarea_size;
};

class cufftContext {
 public:
  cufftContext(cufftPlan* plan);
  ~cufftContext();

 public:
  cufftContext(const cufftContext&)            = delete;
  cufftContext& operator=(const cufftContext&) = delete;

 public:
  cufftContext(cufftContext&&)            = default;
  cufftContext& operator=(cufftContext&&) = default;

 public:
  cufftHandle handle();
  size_t workareaSize();
  void setCallback(cufftXtCallbackType type, void* callback, void* data);

 private:
  cufftPlan* plan_{nullptr};
  std::vector<cufftXtCallbackType> callback_types_{};
};

// Defined in cudalibs.cu

// Return a cached stream for the current GPU
legate::cuda::StreamView get_cached_stream();
cublasHandle_t get_cublas();
cusolverDnHandle_t get_cusolver();
cutensorHandle_t* get_cutensor();
cufftContext get_cufft_plan(cufftType type, const legate::DomainPoint& size);

__host__ inline void check_cublas(cublasStatus_t status, const char* file, int line)
{
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,
            "Internal cuBLAS failure with error code %d in file %s at line %d\n",
            status,
            file,
            line);
#ifdef DEBUG_CUNUMERIC
    assert(false);
#else
    exit(status);
#endif
  }
}

__host__ inline void check_cufft(cufftResult result, const char* file, int line)
{
  if (result != CUFFT_SUCCESS) {
    fprintf(stderr,
            "Internal cuFFT failure with error code %d in file %s at line %d\n",
            result,
            file,
            line);
#ifdef DEBUG_CUNUMERIC
    assert(false);
#else
    exit(result);
#endif
  }
}

__host__ inline void check_cusolver(cusolverStatus_t status, const char* file, int line)
{
  if (status != CUSOLVER_STATUS_SUCCESS) {
    fprintf(stderr,
            "Internal cuSOLVER failure with error code %d in file %s at line %d\n",
            status,
            file,
            line);
#ifdef DEBUG_CUNUMERIC
    assert(false);
#else
    exit(status);
#endif
  }
}

__host__ inline void check_cutensor(cutensorStatus_t result, const char* file, int line)
{
  if (result != CUTENSOR_STATUS_SUCCESS) {
    fprintf(stderr,
            "Internal Legate CUTENSOR failure with error %s (%d) in file %s at line %d\n",
            cutensorGetErrorString(result),
            result,
            file,
            line);
#ifdef DEBUG_CUNUMERIC
    assert(false);
#else
    exit(result);
#endif
  }
}

__host__ inline void check_nccl(ncclResult_t error, const char* file, int line)
{
  if (error != ncclSuccess) {
    fprintf(stderr,
            "Internal NCCL failure with error %s in file %s at line %d\n",
            ncclGetErrorString(error),
            file,
            line);
#ifdef DEBUG_CUNUMERIC
    assert(false);
#else
    exit(error);
#endif
  }
}

template <typename T>
__device__ __forceinline__ T shuffle(unsigned mask, T var, int laneMask, int width)
{
  // return __shfl_xor_sync(0xffffffff, value, i, 32);
  int array[(sizeof(T) + sizeof(int) - 1) / sizeof(int)];
  memcpy(array, &var, sizeof(T));
  for (int& value : array) {
    const int tmp = __shfl_xor_sync(mask, value, laneMask, width);
    value         = tmp;
  }
  memcpy(&var, array, sizeof(T));
  return var;
}

template <typename T>
struct HasNativeShuffle {
  static constexpr bool value = true;
};

template <typename T>
struct HasNativeShuffle<complex<T>> {
  static constexpr bool value = false;
};

template <typename T>
struct HasNativeShuffle<Argval<T>> {
  static constexpr bool value = false;
};

template <typename T, typename REDUCTION>
__device__ __forceinline__ void reduce_output(DeviceScalarReductionBuffer<REDUCTION> result,
                                              T value)
{
  __shared__ T trampoline[THREADS_PER_BLOCK / 32];
  // Reduce across the warp
  const int laneid = threadIdx.x & 0x1f;
  const int warpid = threadIdx.x >> 5;
  for (int i = 16; i >= 1; i /= 2) {
    T shuffle_value;
    if constexpr (HasNativeShuffle<T>::value)
      shuffle_value = __shfl_xor_sync(0xffffffff, value, i, 32);
    else
      shuffle_value = shuffle(0xffffffff, value, i, 32);
    REDUCTION::template fold<true /*exclusive*/>(value, shuffle_value);
  }
  // Write warp values into shared memory
  if ((laneid == 0) && (warpid > 0)) trampoline[warpid] = value;
  __syncthreads();
  // Output reduction
  if (threadIdx.x == 0) {
    for (int i = 1; i < (THREADS_PER_BLOCK / 32); i++)
      REDUCTION::template fold<true /*exclusive*/>(value, trampoline[i]);
    result.reduce<false /*EXCLUSIVE*/>(value);
    // Make sure the result is visible externally
    __threadfence_system();
  }
}

template <typename T>
__device__ __forceinline__ T load_streaming(const T* ptr)
{
  return *ptr;
}

// Specializations to use PTX cache qualifiers to prevent
// loads from interfering with cache state
// Use .cs qualifier to mark the line as evict-first
template <>
__device__ __forceinline__ uint16_t load_streaming<uint16_t>(const uint16_t* ptr)
{
  uint16_t value;
  asm volatile("ld.global.cs.u16 %0, [%1];" : "=h"(value) : "l"(ptr) : "memory");
  return value;
}

template <>
__device__ __forceinline__ uint32_t load_streaming<uint32_t>(const uint32_t* ptr)
{
  uint32_t value;
  asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
  return value;
}

template <>
__device__ __forceinline__ uint64_t load_streaming<uint64_t>(const uint64_t* ptr)
{
  uint64_t value;
  asm volatile("ld.global.cs.u64 %0, [%1];" : "=l"(value) : "l"(ptr) : "memory");
  return value;
}

template <>
__device__ __forceinline__ int16_t load_streaming<int16_t>(const int16_t* ptr)
{
  int16_t value;
  asm volatile("ld.global.cs.s16 %0, [%1];" : "=h"(value) : "l"(ptr) : "memory");
  return value;
}

template <>
__device__ __forceinline__ int32_t load_streaming<int32_t>(const int32_t* ptr)
{
  int32_t value;
  asm volatile("ld.global.cs.s32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
  return value;
}

template <>
__device__ __forceinline__ int64_t load_streaming<int64_t>(const int64_t* ptr)
{
  int64_t value;
  asm volatile("ld.global.cs.s64 %0, [%1];" : "=l"(value) : "l"(ptr) : "memory");
  return value;
}

// No half because inline ptx is dumb about the type

template <>
__device__ __forceinline__ float load_streaming<float>(const float* ptr)
{
  float value;
  asm volatile("ld.global.cs.f32 %0, [%1];" : "=f"(value) : "l"(ptr) : "memory");
  return value;
}

template <>
__device__ __forceinline__ double load_streaming<double>(const double* ptr)
{
  double value;
  asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(value) : "l"(ptr) : "memory");
  return value;
}

template <typename T>
__device__ __forceinline__ void store_streaming(T* ptr, T value)
{
  *ptr = value;
}

// Specializations to use PTX cache qualifiers to avoid
// invalidating read data from caches as we are writing
// Use .cs qualifier to evict first at all levels
template <>
__device__ __forceinline__ void store_streaming<uint16_t>(uint16_t* ptr, uint16_t value)
{
  asm volatile("st.global.cs.u16 [%0], %1;" : : "l"(ptr), "h"(value) : "memory");
}

template <>
__device__ __forceinline__ void store_streaming<uint32_t>(uint32_t* ptr, uint32_t value)
{
  asm volatile("st.global.cs.u32 [%0], %1;" : : "l"(ptr), "r"(value) : "memory");
}

template <>
__device__ __forceinline__ void store_streaming<uint64_t>(uint64_t* ptr, uint64_t value)
{
  asm volatile("st.global.cs.u64 [%0], %1;" : : "l"(ptr), "l"(value) : "memory");
}

template <>
__device__ __forceinline__ void store_streaming<int16_t>(int16_t* ptr, int16_t value)
{
  asm volatile("st.global.cs.s16 [%0], %1;" : : "l"(ptr), "h"(value) : "memory");
}

template <>
__device__ __forceinline__ void store_streaming<int32_t>(int32_t* ptr, int32_t value)
{
  asm volatile("st.global.cs.s32 [%0], %1;" : : "l"(ptr), "r"(value) : "memory");
}

template <>
__device__ __forceinline__ void store_streaming<int64_t>(int64_t* ptr, int64_t value)
{
  asm volatile("st.global.cs.s64 [%0], %1;" : : "l"(ptr), "l"(value) : "memory");
}

// No half because inline ptx is dumb about the type

template <>
__device__ __forceinline__ void store_streaming<float>(float* ptr, float value)
{
  asm volatile("st.global.cs.f32 [%0], %1;" : : "l"(ptr), "f"(value) : "memory");
}

template <>
__device__ __forceinline__ void store_streaming<double>(double* ptr, double value)
{
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(ptr), "d"(value) : "memory");
}

}  // namespace cunumeric
