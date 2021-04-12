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

#include "cuda_help.h"
#include "fill.cuh"
#include "norm.h"
#include "proj.h"
#include "sum.h"
#include <type_traits>

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_vec_norm_2d(const AccessorRW<T, 2> inout, const AccessorRO<T, 2> in, const Rect<2> bounds, const T identity,
                       const int axis, const int order) {
  coord_t        y = bounds.lo[1] + blockIdx.x * blockDim.x + threadIdx.x;
  coord_t        x = bounds.lo[0] + (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y + threadIdx.y;
  const Point<2> p(x, y);
  if (!bounds.contains(p)) return;
  T value = identity;
  if (axis == 0) {
    while (x <= bounds.hi[0]) {
      T val = in[x][y];
      if (std::is_signed<T>::value && val < T(0)) val = -val;
      if (order == 1)
        SumReduction<T>::template fold<true /*exclusive*/>(value, val);
      else if (order == 2) {
        ProdReduction<T>::template fold<true /*exclusive*/>(val, val);
        SumReduction<T>::template fold<true /*exclusive*/>(value, val);
      } else {
        T prod = val;
        for (int i = 0; i < (order - 1); i++)
          ProdReduction<T>::template fold<true /*exclusive*/>(prod, val);
        SumReduction<T>::template fold<true /*exclusive*/>(value, prod);
      }
      x += gridDim.z * gridDim.y * blockDim.y;
    }
  } else {
    while (y <= bounds.hi[1]) {
      T val = in[x][y];
      if (std::is_signed<T>::value && val < T(0)) val = -val;
      if (order == 1)
        SumReduction<T>::template fold<true /*exclusive*/>(value, val);
      else if (order == 2) {
        ProdReduction<T>::template fold<true /*exclusive*/>(val, val);
        SumReduction<T>::template fold<true /*exclusive*/>(value, val);
      } else {
        T prod = val;
        for (int i = 0; i < (order - 1); i++)
          ProdReduction<T>::template fold<true /*exclusive*/>(prod, val);
        SumReduction<T>::template fold<true /*exclusive*/>(value, prod);
      }
      y += gridDim.x * blockDim.x;
    }
#if __CUDA_ARCH__ >= 700
    __shared__ T trampoline[THREADS_PER_BLOCK];
    // Check for the case where all the threads in the same warp have
    // the same x value in which case they're all going to conflict
    // so instead we do a warp-level reduction so just one thread ends
    // up doing the full atomic
    const int same_mask = __match_any_sync(0xffffffff, threadIdx.y);
    int       laneid;
    asm volatile("mov.s32 %0, %laneid;" : "=r"(laneid));
    const int active_mask = __ballot_sync(0xffffffff, same_mask - (1 << laneid));
    if (active_mask) {
      // Store our data into shared
      const int tid   = threadIdx.y * blockDim.x + threadIdx.x;
      trampoline[tid] = value;
      // Make sure all the threads in the warp are done writing
      __syncwarp(active_mask);
      // Have the lowest thread in each mask pull in the values
      int lowest_index = -1;
      for (int i = 0; i < warpSize; i++)
        if (same_mask & (1 << i)) {
          if (lowest_index == -1) {
            if (i != laneid) {
              // We're not the lowest thread in the warp for
              // this value so we're done, set the value back
              // to identity to ensure that we don't try to
              // perform the reduction out to memory
              value = identity;
              break;
            } else    // Make sure we don't do this test again
              lowest_index = i;
            // It was already our value, so just keep going
          } else {
            // Pull in the value from shared memory
            const int index = tid + i - laneid;
            SumReduction<T>::template fold<true /*exclusive*/>(value, trampoline[index]);
          }
        }
    }
#endif
  }
  if (value != identity) SumReduction<T>::template fold<false /*exclusive*/>(inout[p], value);
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_vec_norm_3d(const AccessorRW<T, 3> inout, const AccessorRO<T, 3> in, const Rect<3> bounds, const T identity,
                       const int axis, const int order) {
  coord_t        z = bounds.lo[2] + blockIdx.x * blockDim.x + threadIdx.x;
  coord_t        y = bounds.lo[1] + blockIdx.y * blockDim.y + threadIdx.y;
  coord_t        x = bounds.lo[0] + blockIdx.z * blockDim.z + threadIdx.z;
  const Point<3> p(x, y, z);
  if (!bounds.contains(p)) return;
  T value = identity;
  if (axis == 0) {
    while (x <= bounds.hi[0]) {
      T val = in[x][y][z];
      if (std::is_signed<T>::value && val < T(0)) val = -val;
      if (order == 1)
        SumReduction<T>::template fold<true /*exclusive*/>(value, val);
      else if (order == 2) {
        ProdReduction<T>::template fold<true /*exclusive*/>(val, val);
        SumReduction<T>::template fold<true /*exclusive*/>(value, val);
      } else {
        T prod = val;
        for (int i = 0; i < (order - 1); i++)
          ProdReduction<T>::template fold<true /*exclusive*/>(prod, val);
        SumReduction<T>::template fold<true /*exclusive*/>(value, prod);
      }
      x += gridDim.z * blockDim.z;
    }
  } else if (axis == 1) {
    while (y <= bounds.hi[1]) {
      T val = in[x][y][z];
      if (std::is_signed<T>::value && val < T(0)) val = -val;
      if (order == 1)
        SumReduction<T>::template fold<true /*exclusive*/>(value, val);
      else if (order == 2) {
        ProdReduction<T>::template fold<true /*exclusive*/>(val, val);
        SumReduction<T>::template fold<true /*exclusive*/>(value, val);
      } else {
        T prod = val;
        for (int i = 0; i < (order - 1); i++)
          ProdReduction<T>::template fold<true /*exclusive*/>(prod, val);
        SumReduction<T>::template fold<true /*exclusive*/>(value, prod);
      }
      y += gridDim.y * blockDim.y;
    }
  } else {
    while (z <= bounds.hi[2]) {
      T val = in[x][y][z];
      if (std::is_signed<T>::value && val < T(0)) val = -val;
      if (order == 1)
        SumReduction<T>::template fold<true /*exclusive*/>(value, val);
      else if (order == 2) {
        ProdReduction<T>::template fold<true /*exclusive*/>(val, val);
        SumReduction<T>::template fold<true /*exclusive*/>(value, val);
      } else {
        T prod = val;
        for (int i = 0; i < (order - 1); i++)
          ProdReduction<T>::template fold<true /*exclusive*/>(prod, val);
        SumReduction<T>::template fold<true /*exclusive*/>(value, prod);
      }
      z += gridDim.x * blockDim.x;
    }
#if __CUDA_ARCH__ >= 700
    __shared__ T trampoline[THREADS_PER_BLOCK];
    // Check for the case where all the threads in the same warp have
    // the same x value in which case they're all going to conflict
    // so instead we do a warp-level reduction so just one thread ends
    // up doing the full atomic
    const int same_mask = __match_any_sync(0xffffffff, threadIdx.z * blockDim.y + threadIdx.y);
    int       laneid;
    asm volatile("mov.s32 %0, %laneid;" : "=r"(laneid));
    const int active_mask = __ballot_sync(0xffffffff, same_mask - (1 << laneid));
    if (active_mask) {
      // Store our data into shared
      const int tid   = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
      trampoline[tid] = value;
      // Make sure all the threads in the warp are done writing
      __syncwarp(active_mask);
      // Have the lowest thread in each mask pull in the values
      int lowest_index = -1;
      for (int i = 0; i < warpSize; i++)
        if (same_mask & (1 << i)) {
          if (lowest_index == -1) {
            if (i != laneid) {
              // We're not the lowest thread in the warp for
              // this value so we're done, set the value back
              // to identity to ensure that we don't try to
              // perform the reduction out to memory
              value = identity;
              break;
            } else    // Make sure we don't do this test again
              lowest_index = i;
            // It was already our value, so just keep going
          } else {
            // Pull in the value from shared memory
            const int index = tid + i - laneid;
            SumReduction<T>::template fold<true /*exclusive*/>(value, trampoline[index]);
          }
        }
    }
#endif
  }
  if (value != identity) SumReduction<T>::template fold<false /*exclusive*/>(inout[p], value);
}

template<typename T>
/*static*/ void NormTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                         Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          axis         = derez.unpack_dimension();
  const int          collapse_dim = derez.unpack_dimension();
  const int          init_dim     = derez.unpack_dimension();
  switch (init_dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<T, 1> out =
          (collapse_dim >= 0) ? derez.unpack_accessor_WO<T, 1>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                              : derez.unpack_accessor_WO<T, 1>(regions[0], rect);
      const size_t volume = rect.volume();
      const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      legate_fill_1d<T><<<blocks, THREADS_PER_BLOCK>>>(out, SumReduction<T>::identity, rect.lo, volume);
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<T, 2> out =
          (collapse_dim >= 0) ? derez.unpack_accessor_WO<T, 2>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                              : derez.unpack_accessor_WO<T, 2>(regions[0], rect);
      const size_t  volume = rect.volume();
      const size_t  blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t pitch  = rect.hi[1] - rect.lo[1] + 1;
      legate_fill_2d<T><<<blocks, THREADS_PER_BLOCK>>>(out, SumReduction<T>::identity, rect.lo, Point<1>(pitch), volume);
      break;
    }
    default:
      assert(false);    // shouldn't see any other cases
  }
  const int dim   = derez.unpack_dimension();
  const int order = task->futures[0].get_result<int>();
  assert(order > 0);
  switch (dim) {
    // Should never get the case of 1 as this would just be a copy since
    // reducing our only dimension should have called SumReducTask
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorRW<T, 2> inout =
          (collapse_dim >= 0) ? derez.unpack_accessor_RW<T, 2, 1>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                              : derez.unpack_accessor_RW<T, 2>(regions[0], rect);
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      // Figure out how many blocks and threads we need
      dim3 threads(1, 1, 1);
      dim3 blocks(1, 1, 1);
      raster_2d_reduction(blocks, threads, rect, axis, (const void*)legate_vec_norm_2d<T>);
      legate_vec_norm_2d<T><<<blocks, threads>>>(inout, in, rect, SumReduction<T>::identity, axis, order);
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) return;
      const AccessorRW<T, 3> inout =
          (collapse_dim >= 0) ? derez.unpack_accessor_RW<T, 3, 2>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                              : derez.unpack_accessor_RW<T, 3>(regions[0], rect);
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      // Figure out how many blocks and threads we need
      dim3 threads(1, 1, 1);
      dim3 blocks(1, 1, 1);
      raster_3d_reduction(blocks, threads, rect, axis, (const void*)legate_vec_norm_3d<T>);
      legate_vec_norm_3d<T><<<blocks, threads>>>(inout, in, rect, SumReduction<T>::identity, axis, order);
      break;
    }
    default:
      assert(false);
  }
}

INSTANTIATE_TASK_VARIANT(NormTask, gpu_variant)

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_vec_norm_reduce_1d(DeferredReduction<SumReduction<T>> result, const AccessorRO<T, 1> in, const size_t iters,
                              const Point<1> origin, const size_t max, const int order, const T identity) {
  T value = identity;
  if (order == 1) {
    for (unsigned idx = 0; idx < iters; idx++) {
      const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
      if (offset < max) {
        const coord_t x   = origin[0] + offset;
        T             val = in[x];
        if (std::is_signed<T>::value && val < T(0)) val = -val;
        SumReduction<T>::template fold<true>(value, val);
      }
    }
  } else if (order == 2) {
    for (unsigned idx = 0; idx < iters; idx++) {
      const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
      if (offset < max) {
        const coord_t x   = origin[0] + offset;
        T             val = in[x];
        ProdReduction<T>::template fold<true>(val, val);
        SumReduction<T>::template fold<true>(value, val);
      }
    }
  } else {
    for (unsigned idx = 0; idx < iters; idx++) {
      const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
      if (offset < max) {
        const coord_t x   = origin[0] + offset;
        T             val = in[x];
        if (std::is_signed<T>::value && val < T(0)) val = -val;
        T prod = val;
        for (int i = 0; i < (order - 1); i++)
          ProdReduction<T>::template fold<true>(prod, val);
        SumReduction<T>::template fold<true>(value, prod);
      }
    }
  }
  reduce_output(result, value);
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_vec_norm_reduce_2d(DeferredReduction<SumReduction<T>> result, const AccessorRO<T, 2> in, const size_t iters,
                              const Point<2> origin, const Point<1> pitch, const size_t max, const int order, const T identity) {
  T value = identity;
  if (order == 1) {
    for (unsigned idx = 0; idx < iters; idx++) {
      const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
      if (offset < max) {
        const coord_t x   = origin[0] + offset / pitch[0];
        const coord_t y   = origin[1] + offset % pitch[0];
        T             val = in[x][y];
        if (std::is_signed<T>::value && val < T(0)) val = -val;
        SumReduction<T>::template fold<true>(value, val);
      }
    }
  } else if (order == 2) {
    for (unsigned idx = 0; idx < iters; idx++) {
      const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
      if (offset < max) {
        const coord_t x   = origin[0] + offset / pitch[0];
        const coord_t y   = origin[1] + offset % pitch[0];
        T             val = in[x][y];
        ProdReduction<T>::template fold<true>(val, val);
        SumReduction<T>::template fold<true>(value, val);
      }
    }
  } else {
    for (unsigned idx = 0; idx < iters; idx++) {
      const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
      if (offset < max) {
        const coord_t x   = origin[0] + offset / pitch[0];
        const coord_t y   = origin[1] + offset % pitch[0];
        T             val = in[x][y];
        if (std::is_signed<T>::value && val < T(0)) val = -val;
        T prod = val;
        for (int i = 0; i < (order - 1); i++)
          ProdReduction<T>::template fold<true>(prod, val);
        SumReduction<T>::template fold<true>(value, prod);
      }
    }
  }
  reduce_output(result, value);
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_vec_norm_reduce_3d(DeferredReduction<SumReduction<T>> result, const AccessorRO<T, 3> in, const size_t iters,
                              const Point<3> origin, const Point<2> pitch, const size_t max, const int order, const T identity) {
  T value = identity;
  if (order == 1) {
    for (unsigned idx = 0; idx < iters; idx++) {
      const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
      if (offset < max) {
        const coord_t x   = origin[0] + offset / pitch[0];
        const coord_t y   = origin[1] + (offset % pitch[0]) / pitch[1];
        const coord_t z   = origin[2] + (offset % pitch[0]) % pitch[1];
        T             val = in[x][y][z];
        if (std::is_signed<T>::value && val < T(0)) val = -val;
        SumReduction<T>::template fold<true>(value, val);
      }
    }
  } else if (order == 2) {
    for (unsigned idx = 0; idx < iters; idx++) {
      const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
      if (offset < max) {
        const coord_t x   = origin[0] + offset / pitch[0];
        const coord_t y   = origin[1] + (offset % pitch[0]) / pitch[1];
        const coord_t z   = origin[2] + (offset % pitch[0]) % pitch[1];
        T             val = in[x][y][z];
        ProdReduction<T>::template fold<true>(val, val);
        SumReduction<T>::template fold<true>(value, val);
      }
    }
  } else {
    for (unsigned idx = 0; idx < iters; idx++) {
      const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
      if (offset < max) {
        const coord_t x   = origin[0] + offset / pitch[0];
        const coord_t y   = origin[1] + (offset % pitch[0]) / pitch[1];
        const coord_t z   = origin[2] + (offset % pitch[0]) % pitch[1];
        T             val = in[x][y][z];
        if (std::is_signed<T>::value && val < T(0)) val = -val;
        T prod = val;
        for (int i = 0; i < (order - 1); i++)
          ProdReduction<T>::template fold<true>(prod, val);
        SumReduction<T>::template fold<true>(value, prod);
      }
    }
  }
  reduce_output(result, value);
}

template<typename T>
/*static*/ DeferredReduction<SumReduction<T>>
    NormReducTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim   = derez.unpack_dimension();
  const int          order = task->futures[0].get_result<int>();
  assert(order > 0);
  DeferredReduction<SumReduction<T>> result;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in     = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
      const size_t           volume = rect.volume();
      const size_t           blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      if (blocks >= MAX_REDUCTION_CTAS) {
        const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
        legate_vec_norm_reduce_1d<T>
            <<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK>>>(result, in, iters, rect.lo, volume, order, SumReduction<T>::identity);
      } else {
        legate_vec_norm_reduce_1d<T>
            <<<blocks, THREADS_PER_BLOCK>>>(result, in, 1 /*iters*/, rect.lo, volume, order, SumReduction<T>::identity);
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2> in     = derez.unpack_accessor_RO<T, 2>(regions[0], rect);
      const size_t           volume = rect.volume();
      const size_t           blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t          pitch  = rect.hi[1] - rect.lo[1] + 1;
      if (blocks >= MAX_REDUCTION_CTAS) {
        const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
        legate_vec_norm_reduce_2d<T><<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK>>>(result, in, iters, rect.lo, Point<1>(pitch), volume,
                                                                                order, SumReduction<T>::identity);
      } else {
        legate_vec_norm_reduce_2d<T><<<blocks, THREADS_PER_BLOCK>>>(result, in, 1 /*iter*/, rect.lo, Point<1>(pitch), volume, order,
                                                                    SumReduction<T>::identity);
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3> in       = derez.unpack_accessor_RO<T, 3>(regions[0], rect);
      const size_t           volume   = rect.volume();
      const size_t           blocks   = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t          diffy    = rect.hi[1] - rect.lo[1] + 1;
      const coord_t          diffz    = rect.hi[2] - rect.lo[2] + 1;
      const coord_t          pitch[2] = {diffy * diffz, diffz};
      if (blocks >= MAX_REDUCTION_CTAS) {
        const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
        legate_vec_norm_reduce_3d<T><<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK>>>(result, in, iters, rect.lo, Point<2>(pitch), volume,
                                                                                order, SumReduction<T>::identity);
      } else {
        legate_vec_norm_reduce_3d<T><<<blocks, THREADS_PER_BLOCK>>>(result, in, 1 /*iter*/, rect.lo, Point<2>(pitch), volume, order,
                                                                    SumReduction<T>::identity);
      }
      break;
    }
    default:
      assert(false);    // should have any other dimensions
  }
  return result;
}

INSTANTIATE_DEFERRED_REDUCTION_TASK_VARIANT(NormReducTask, SumReduction, gpu_variant)

}    // namespace numpy
}    // namespace legate
