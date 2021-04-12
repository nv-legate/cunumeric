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
#  define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif
#ifndef MIN
#  define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

namespace legate {
namespace numpy {

__host__ inline void check_cublas(cublasStatus_t status, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "Internal Legate CUBLAS failure with error code %d in file %s at line %d\n", status, file, line);
    exit(status);
  }
}

__host__ inline void raster_2d_reduction(dim3& blocks, dim3& threads, const Legion::Rect<2> rect, const int axis,
                                         const void* func) {
  if (axis == 0) {
    // Put as many threads as possible on the non-collapsing dimension
    const size_t non_collapse_size = (rect.hi[1] - rect.lo[1]) + 1;
    // Transpose thread dimensions for warp goodness
    threads.x = MIN(non_collapse_size, THREADS_PER_BLOCK);
    ;
    if (threads.x < THREADS_PER_BLOCK) threads.y = MAX(THREADS_PER_BLOCK / threads.x, 1);
  } else {
    // Put at least 32 threads on the last dimension since
    // we still want warp coalescing for bandwidth reasons
    // Transpose thread dimensions for warp goodness
    threads.x = MIN((rect.hi[1] - rect.lo[1]) + 1, 32);
    if (threads.x < THREADS_PER_BLOCK) threads.y = MAX(THREADS_PER_BLOCK / threads.x, 1);
  }
  // Have number of threads per block, figure out how many CTAs we can fit
  // on the GPU to make sure we fill it up, but we want the minimum number
  // fill it up so we can walk as long as possible
  int num_ctas = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_ctas, func, threads.x * threads.y, 0);
  if (axis == 0) {
    // Transpose CTA dimensions for consistency too
    // Strip CTAs across the non-collapsing dimension first
    blocks.x = ((rect.hi[1] - rect.lo[1]) + threads.x) / threads.x;
    // If we didn't get enough CTAs that way, fill them as little as
    // possible along the collapsing dimension
    if (blocks.x < num_ctas) {
      blocks.y = MIN((num_ctas + blocks.x - 1) / blocks.x, ((rect.hi[0] - rect.lo[0]) + threads.y) / threads.y);
      // Handle CUDA boundary problem
      while (blocks.y > 65536) {
        assert((blocks.y % 2) == 0);
        blocks.y /= 2;
        blocks.z *= 2;
      }
    }
  } else {
    // Transpose CTA dimensions for consistency too
    // Strip CTAs across the non-collapsing dimension first
    blocks.y = ((rect.hi[0] - rect.lo[0]) + threads.y) / threads.y;
    // If we didn't get enough CTAs that way, fill them as little as
    // possible along the collapsing dimension
    if (blocks.y < num_ctas)
      blocks.x = MIN((num_ctas + blocks.y - 1) / blocks.y, ((rect.hi[1] - rect.lo[1]) + threads.x) / threads.x);
    // Handle CUDA boundary problem
    while (blocks.y > 65536) {
      assert((blocks.y % 2) == 0);
      blocks.y /= 2;
      blocks.z *= 2;
    }
  }
}

__host__ inline void raster_3d_reduction(dim3& blocks, dim3& threads, const Legion::Rect<3> rect, const int axis,
                                         const void* func) {
  if (axis == 0) {
    // Transpose thread dimensions for warp goodness
    threads.x = MIN((rect.hi[2] - rect.lo[2]) + 1, THREADS_PER_BLOCK);
    if (threads.x < THREADS_PER_BLOCK) {
      const size_t remainder_threads = (THREADS_PER_BLOCK + threads.x - 1) / threads.x;
      threads.y                      = MIN((rect.hi[1] - rect.lo[1]) + 1, remainder_threads);
      if ((threads.x * threads.y) < THREADS_PER_BLOCK) threads.z = MAX(THREADS_PER_BLOCK / (threads.x * threads.y), 1);
    }
  } else if (axis == 1) {
    // Transpose thread dimensions for warp goodness
    threads.x = MIN((rect.hi[2] - rect.lo[2]) + 1, THREADS_PER_BLOCK);
    if (threads.x < THREADS_PER_BLOCK) {
      const size_t remainder_threads = (THREADS_PER_BLOCK + threads.x - 1) / threads.x;
      threads.z                      = MIN((rect.hi[0] - rect.lo[0]) + 1, remainder_threads);
      if ((threads.x * threads.z) < THREADS_PER_BLOCK) threads.y = MAX(THREADS_PER_BLOCK / (threads.x * threads.z), 1);
    }
  } else {
    assert(axis == 2);
    // Put at least 32 threads on the last dimension since we still
    // want warp coalescing for bandwidth reasons
    // Transpose thread dimensions for warp goodness
    threads.x = MIN((rect.hi[2] - rect.lo[2]) + 1, 32);
    if (threads.x < THREADS_PER_BLOCK) {
      const size_t remainder_threads = (THREADS_PER_BLOCK + threads.x - 1) / threads.x;
      threads.y                      = MIN((rect.hi[1] - rect.lo[1]) + 1, remainder_threads);
      if ((threads.x * threads.y) < THREADS_PER_BLOCK) threads.z = MAX(THREADS_PER_BLOCK / (threads.x * threads.y), 1);
    }
  }
  // Have number of threads per block, figure out how many CTAs we can fit
  // on the GPU to make sure we fill it up, but we want the minimum number
  // fill it up so we can walk as long as possible
  int num_ctas = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_ctas, func, threads.x * threads.y * threads.z, 0);
  if (axis == 0) {
    // Transpose CTA dimensions for consistency too
    blocks.y = ((rect.hi[1] - rect.lo[1]) + threads.y) / threads.y;
    blocks.x = ((rect.hi[2] - rect.lo[2]) + threads.x) / threads.x;
    if ((blocks.x * blocks.y) < num_ctas)
      blocks.z =
          MIN((num_ctas + (blocks.x * blocks.y) - 1) / (blocks.x * blocks.y), ((rect.hi[0] - rect.lo[0]) + threads.z) / threads.z);
  } else if (axis == 1) {
    blocks.z = ((rect.hi[0] - rect.lo[0]) + threads.z) / threads.z;
    blocks.x = ((rect.hi[2] - rect.lo[2]) + threads.x) / threads.x;
    if ((blocks.x * blocks.z) < num_ctas)
      blocks.y =
          MIN((num_ctas + (blocks.x * blocks.z) - 1) / (blocks.x * blocks.z), ((rect.hi[1] - rect.lo[1]) + threads.y) / threads.y);
  } else {
    blocks.z = ((rect.hi[0] - rect.lo[0]) + threads.z) / threads.z;
    blocks.y = ((rect.hi[1] - rect.lo[1]) + threads.y) / threads.y;
    if ((blocks.y * blocks.z) < num_ctas)
      blocks.x =
          MIN((num_ctas + (blocks.y * blocks.z) - 1) / (blocks.y * blocks.z), ((rect.hi[2] - rect.lo[2]) + threads.x) / threads.x);
  }
  // CUDA boundary checks in case we hit them
  // TODO: if we hit one of these assertions fix them in a way
  // similar to what we did for 2D cases above
  assert(blocks.y <= 65536);
  assert(blocks.z <= 65536);
}

template<typename T>
__device__ __forceinline__ T shuffle(unsigned mask, T var, int laneMask, int width) {
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
// TBD: if compiler optimizes out the shuffle function we defined, we could make it the default version
template<typename T, typename REDUCTION>
__device__ __forceinline__ void reduce_output(Legion::DeferredReduction<REDUCTION> result, complex<T> value) {
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

template<typename T, typename REDUCTION>
__device__ __forceinline__ void reduce_output(Legion::DeferredReduction<REDUCTION> result, T value) {
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

__device__ __forceinline__ void reduce_bool(Legion::DeferredValue<bool> result, int value) {
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

template<typename REDUCTION, typename T>
__device__ __forceinline__ void fold_output(Legion::DeferredBuffer<T, 1> buffer, T value, const REDUCTION) {
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
    // Write to the block index since there's just one output for this block
    buffer.write(blockIdx.x, value);
  }
}

template<typename REDUCTION, typename T>
__device__ __forceinline__ void fold_output(Legion::DeferredBuffer<complex<T>, 1> buffer, complex<T> value, const REDUCTION) {
#pragma diag_suppress           static_var_with_dynamic_init
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
    // Write to the block index since there's just one output for this block
    buffer.write(blockIdx.x, value);
  }
}

__device__ __forceinline__ void fold_bool(Legion::DeferredBuffer<bool, 1> result, int value) {
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
    // Write to the block index since there's just one output for this block
    if (value == 0)
      result.write(blockIdx.x, false);
    else
      result.write(blockIdx.x, true);
  }
}

}    // namespace numpy
}    // namespace legate

#endif    // __CUDA_HELP_H__
