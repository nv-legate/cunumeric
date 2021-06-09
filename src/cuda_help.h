/* Copyright 2021 NVIDIA Corporation
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

#ifndef __CUDA_HELP_H__
#define __CUDA_HELP_H__

#include "legate.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 128
#define MIN_CTAS_PER_SM 4
#define MAX_REDUCTION_CTAS 1024

#define CHECK_CUBLAS(expr)                    \
  {                                           \
    cublasStatus_t result = (expr);           \
    check_cublas(result, __FILE__, __LINE__); \
  }

#ifndef MAX
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif
#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

namespace legate {
namespace numpy {

__host__ inline void check_cublas(cublasStatus_t status, const char* file, int line)
{
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,
            "Internal Legate CUBLAS failure with error code %d in file %s at line %d\n",
            status,
            file,
            line);
    exit(status);
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

// Overload for complex
// TBD: if compiler optimizes out the shuffle function we defined, we could make it the default
// version
template <typename T, typename REDUCTION>
__device__ __forceinline__ void reduce_output(Legion::DeferredReduction<REDUCTION> result,
                                              complex<T> value)
{
  __shared__ complex<T> trampoline[THREADS_PER_BLOCK / 32];
  // Reduce across the warp
  const int laneid = threadIdx.x & 0x1f;
  const int warpid = threadIdx.x >> 5;
  for (int i = 16; i >= 1; i /= 2) {
    const complex<T> shuffle_value = shuffle(0xffffffff, value, i, 32);
    REDUCTION::template fold<true /*exclusive*/>(value, shuffle_value);
  }
  // Write warp values into shared memory
  if ((laneid == 0) && (warpid > 0)) trampoline[warpid] = value;
  __syncthreads();
  // Output reduction
  if (threadIdx.x == 0) {
    for (int i = 1; i < (THREADS_PER_BLOCK / 32); i++)
      REDUCTION::template fold<true /*exclusive*/>(value, trampoline[i]);
    result <<= value;
    // Make sure the result is visible externally
    __threadfence_system();
  }
}

template <typename T, typename REDUCTION>
__device__ __forceinline__ void reduce_output(Legion::DeferredReduction<REDUCTION> result, T value)
{
  __shared__ T trampoline[THREADS_PER_BLOCK / 32];
  // Reduce across the warp
  const int laneid = threadIdx.x & 0x1f;
  const int warpid = threadIdx.x >> 5;
  for (int i = 16; i >= 1; i /= 2) {
    const T shuffle_value = __shfl_xor_sync(0xffffffff, value, i, 32);
    REDUCTION::template fold<true /*exclusive*/>(value, shuffle_value);
  }
  // Write warp values into shared memory
  if ((laneid == 0) && (warpid > 0)) trampoline[warpid] = value;
  __syncthreads();
  // Output reduction
  if (threadIdx.x == 0) {
    for (int i = 1; i < (THREADS_PER_BLOCK / 32); i++)
      REDUCTION::template fold<true /*exclusive*/>(value, trampoline[i]);
    result <<= value;
    // Make sure the result is visible externally
    __threadfence_system();
  }
}

__device__ __forceinline__ void reduce_bool(Legion::DeferredValue<bool> result, int value)
{
  __shared__ int trampoline[THREADS_PER_BLOCK / 32];
  // Reduce across the warp
  const int laneid = threadIdx.x & 0x1f;
  const int warpid = threadIdx.x >> 5;
  for (int i = 16; i >= 1; i /= 2) {
    const int shuffle_value = __shfl_xor_sync(0xffffffff, value, i, 32);
    if (shuffle_value == 0) value = 0;
  }
  // Write warp values into shared memory
  if ((laneid == 0) && (warpid > 0)) trampoline[warpid] = value;
  __syncthreads();
  // Output reduction
  if (threadIdx.x == 0) {
    for (int i = 1; i < (THREADS_PER_BLOCK / 32); i++)
      if (trampoline[i] == 0) {
        value = 0;
        break;
      }
    if (value == 0) {
      result = false;
      // Make sure the result is visible externally
      __threadfence_system();
    }
  }
}

}  // namespace numpy
}  // namespace legate

#endif  // __CUDA_HELP_H__
