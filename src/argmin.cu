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

#include "argmin.h"
#include "cuda_help.h"
#include "fill.cuh"
#include "proj.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_argmin_2d(const AccessorRW<Argval<T>, 2> inout, const AccessorRO<T, 2> in, const Rect<2> bounds, const T identity,
                     const int axis) {
  coord_t        y = bounds.lo[1] + blockIdx.x * blockDim.x + threadIdx.x;
  coord_t        x = bounds.lo[0] + (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y + threadIdx.y;
  const Point<2> p(x, y);
#pragma diag_suppress static_var_with_dynamic_init
  __shared__ T values[THREADS_PER_BLOCK];
  __shared__ coord_t indexes[THREADS_PER_BLOCK];
  const int          tid = threadIdx.y * blockDim.x + threadIdx.x;
  Argval<T>          value(identity);
  if (axis == 0) {
    if (bounds.contains(p)) {
      while (x <= bounds.hi[0]) {
        Argval<T> next(x, in[x][y]);
        ArgminReduction<T>::template fold<true /*exclusive*/>(value, next);
        x += gridDim.z * gridDim.y * blockDim.y;
      }
    }
    // Save the results into shared memory
    values[tid]  = value.arg_value;
    indexes[tid] = value.arg;
  } else {
    if (bounds.contains(p)) {
      while (y <= bounds.hi[1]) {
        Argval<T> next(y, in[x][y]);
        ArgminReduction<T>::template fold<true /*exclusive*/>(value, next);
        y += gridDim.x * blockDim.x;
      }
    }
    values[tid]  = value.arg_value;
    indexes[tid] = value.arg;
  }
  // Wait for all the threads to be done
  __syncthreads();
  // TODO: We could make these reductions more warp-sensitive to be faster
  // but we're too lazy to do that right now
  if (axis == 0) {
    if ((threadIdx.y == 0) && bounds.contains(p)) {
      for (int i = 1; i < blockDim.y; i++) {
        const int next_tid = tid + i * blockDim.x;
        Argval<T> next(indexes[next_tid], values[next_tid]);
        ArgminReduction<T>::template fold<true /*exclusive*/>(value, next);
      }
      // Then do the reduction out to memory
      ArgminReduction<T>::template fold<false /*exclusive*/>(inout[p], value);
    }
  } else {
    if ((threadIdx.x == 0) && bounds.contains(p)) {
      for (int i = 1; i < blockDim.x; i++) {
        const int next_tid = tid + i;
        Argval<T> next(indexes[next_tid], values[next_tid]);
        ArgminReduction<T>::template fold<true /*exclusive*/>(value, next);
      }
      // Then do the reduction out to memory
      ArgminReduction<T>::template fold<false /*exclusive*/>(inout[p], value);
    }
  }
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_argmin_3d(const AccessorRW<Argval<T>, 3> inout, const AccessorRO<T, 3> in, const Rect<3> bounds, const T identity,
                     const int axis) {
  coord_t        z = bounds.lo[2] + blockIdx.x * blockDim.x + threadIdx.x;
  coord_t        y = bounds.lo[1] + blockIdx.y * blockDim.y + threadIdx.y;
  coord_t        x = bounds.lo[0] + blockIdx.z * blockDim.z + threadIdx.z;
  const Point<3> p(x, y, z);
  __shared__ T values[THREADS_PER_BLOCK];
  __shared__ coord_t indexes[THREADS_PER_BLOCK];
  const int          tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  Argval<T>          value(identity);
  if (axis == 0) {
    if (bounds.contains(p)) {
      while (x <= bounds.hi[0]) {
        Argval<T> next(x, in[x][y][z]);
        ArgminReduction<T>::template fold<true /*exclusive*/>(value, next);
        x += gridDim.z * blockDim.z;
      }
    }
    // Save the results into shared memory
    values[tid]  = value.arg_value;
    indexes[tid] = value.arg;
  } else if (axis == 1) {
    if (bounds.contains(p)) {
      while (y <= bounds.hi[1]) {
        Argval<T> next(y, in[x][y][z]);
        ArgminReduction<T>::template fold<true /*exclusive*/>(value, next);
        y += gridDim.y * blockDim.y;
      }
    }
    values[tid]  = value.arg_value;
    indexes[tid] = value.arg;
  } else {
    if (bounds.contains(p)) {
      while (z <= bounds.hi[2]) {
        Argval<T> next(z, in[x][y][z]);
        ArgminReduction<T>::template fold<true /*exclusive*/>(value, next);
        z += gridDim.x * blockDim.x;
      }
    }
    values[tid]  = value.arg_value;
    indexes[tid] = value.arg;
  }
  // Wait for all the threads to be done
  __syncthreads();
  // TODO: We could make these reductions more warp-sensitive to be faster
  // but we're too lazy to do that right now
  if (axis == 0) {
    if ((threadIdx.z == 0) && bounds.contains(p)) {
      for (int i = 1; i < blockDim.z; i++) {
        const int next_tid = tid + i * blockDim.y * blockDim.x;
        Argval<T> next(indexes[next_tid], values[next_tid]);
        ArgminReduction<T>::template fold<true /*exclusive*/>(value, next);
      }
      // Then do the reduction out to memory
      ArgminReduction<T>::template fold<false /*exclusive*/>(inout[p], value);
    }
  } else if (axis == 1) {
    if ((threadIdx.y == 0) && bounds.contains(p)) {
      for (int i = 1; i < blockDim.y; i++) {
        const int next_tid = tid + i * blockDim.x;
        Argval<T> next(indexes[next_tid], values[next_tid]);
        ArgminReduction<T>::template fold<true /*exclusive*/>(value, next);
      }
      // Then do the reduction out to memory
      ArgminReduction<T>::template fold<false /*exclusive*/>(inout[p], value);
    }
  } else {
    if ((threadIdx.x == 0) && bounds.contains(p)) {
      for (int i = 0; i < blockDim.x; i++) {
        const int next_tid = tid + i;
        Argval<T> next(indexes[next_tid], values[next_tid]);
        ArgminReduction<T>::template fold<true /*exclusive*/>(value, next);
      }
      // Then do the reduction out to memory
      ArgminReduction<T>::template fold<false /*exclusive*/>(inout[p], value);
    }
  }
}

template<typename T>
/*static*/ void ArgminTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                           Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          axis         = derez.unpack_dimension();
  const int          collapse_dim = derez.unpack_dimension();
  const int          init_dim     = derez.unpack_dimension();
  switch (init_dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<Argval<T>, 1> out =
          (collapse_dim >= 0)
              ? derez.unpack_accessor_WO<Argval<T>, 1>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
              : derez.unpack_accessor_WO<Argval<T>, 1>(regions[0], rect);
      const size_t volume = rect.volume();
      const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      legate_fill_1d<Argval<T>><<<blocks, THREADS_PER_BLOCK>>>(out, ArgminReduction<T>::identity, rect.lo, volume);
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<Argval<T>, 2> out =
          (collapse_dim >= 0)
              ? derez.unpack_accessor_WO<Argval<T>, 2>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
              : derez.unpack_accessor_WO<Argval<T>, 2>(regions[0], rect);
      const size_t  volume = rect.volume();
      const size_t  blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t pitch  = rect.hi[1] - rect.lo[1] + 1;
      legate_fill_2d<Argval<T>><<<blocks, THREADS_PER_BLOCK>>>(out, ArgminReduction<T>::identity, rect.lo, Point<1>(pitch), volume);
      break;
    }
    default:
      assert(false);    // shouldn't see any other cases
  }
  const int dim = derez.unpack_dimension();
  switch (dim) {
    // Should never get the case of 1 as this would just be a copy since
    // reducing our only dimension should have called MinReducTask
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorRW<Argval<T>, 2> inout =
          (collapse_dim >= 0)
              ? derez.unpack_accessor_RW<Argval<T>, 2, 1>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
              : derez.unpack_accessor_RW<Argval<T>, 2>(regions[0], rect);
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      // First figure out how many threads per block we can have
      dim3 threads(1, 1, 1);
      dim3 blocks(1, 1, 1);
      raster_2d_reduction(blocks, threads, rect, axis, (const void*)legate_argmin_2d<T>);
      legate_argmin_2d<T><<<blocks, threads>>>(inout, in, rect, MinReduction<T>::identity, axis);
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) return;
      const AccessorRW<Argval<T>, 3> inout =
          (collapse_dim >= 0)
              ? derez.unpack_accessor_RW<Argval<T>, 3, 2>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
              : derez.unpack_accessor_RW<Argval<T>, 3>(regions[0], rect);
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      // First figure out how many threads per block we can have
      dim3 threads(1, 1, 1);
      dim3 blocks(1, 1, 1);
      raster_3d_reduction(blocks, threads, rect, axis, (const void*)legate_argmin_3d<T>);
      legate_argmin_3d<T><<<blocks, threads>>>(inout, in, rect, MinReduction<T>::identity, axis);
      break;
    }
    default:
      assert(false);
  }
}

INSTANTIATE_TASK_VARIANT(ArgminTask, gpu_variant)

template<typename T>
/*static*/ DeferredReduction<ArgminReduction<T>>
    ArgminReducTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  DeferredReduction<ArgminReduction<T>> result;
  // TODO:
  assert(false);
  return result;
}

INSTANTIATE_DEFERRED_REDUCTION_TASK_VARIANT(ArgminReducTask, ArgminReduction, gpu_variant)

template<typename T, int DIM>
struct ArgminRadixArgs {
  AccessorRO<Argval<T>, DIM> in[MAX_REDUCTION_RADIX];
};

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_argmin_radix_1d(const AccessorWO<Argval<T>, 1> out, const ArgminRadixArgs<T, 1> args, const size_t argmax,
                           const Point<1> origin, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x   = origin[0] + offset;
  Argval<T>     val = args.in[0][x];
  for (unsigned idx = 1; idx < argmax; idx++)
    ArgminReduction<T>::template fold<true /*exclusive*/>(val, args.in[idx][x]);
  out[x] = val;
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_argmin_radix_2d(const AccessorWO<Argval<T>, 2> out, const ArgminRadixArgs<T, 2> args, const size_t argmax,
                           const Point<2> origin, const Point<1> pitch, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x   = origin[0] + offset / pitch[0];
  const coord_t y   = origin[1] + offset % pitch[0];
  Argval<T>     val = args.in[0][x][y];
  for (unsigned idx = 1; idx < argmax; idx++)
    ArgminReduction<T>::template fold<true /*exclusive*/>(val, args.in[idx][x][y]);
  out[x][y] = val;
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_argmin_radix_3d(const AccessorWO<Argval<T>, 3> out, const ArgminRadixArgs<T, 3> args, const size_t argmax,
                           const Point<3> origin, const Point<2> pitch, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x   = origin[0] + offset / pitch[0];
  const coord_t y   = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z   = origin[2] + (offset % pitch[0]) % pitch[1];
  Argval<T>     val = args.in[0][x][y][z];
  for (unsigned idx = 1; idx < argmax; idx++)
    ArgminReduction<T>::template fold<true /*exclusive*/>(val, args.in[idx][x][y][z]);
  out[x][y][z] = val;
}

template<typename T>
/*static*/ void ArgminRadixTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  assert(task->regions.size() <= MAX_REDUCTION_RADIX);
  const int     radix         = derez.unpack_dimension();
  const int     extra_dim_out = derez.unpack_dimension();
  const int     extra_dim_in  = derez.unpack_dimension();
  const int     dim           = derez.unpack_dimension();
  const coord_t offset        = (extra_dim_in >= 0) ? task->index_point[extra_dim_in] * radix : 0;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<Argval<T>, 1> out =
          (extra_dim_out >= 0)
              ? derez.unpack_accessor_WO<Argval<T>, 1>(regions[0], rect, extra_dim_out, task->index_point[extra_dim_out])
              : derez.unpack_accessor_WO<Argval<T>, 1>(regions[0], rect);
      ArgminRadixArgs<T, 1> args;
      unsigned              num_inputs = 0;
      for (unsigned idx = 1; idx < task->regions.size(); idx++)
        if (task->regions[idx].region.exists())
          args.in[num_inputs++] = derez.unpack_accessor_RO<Argval<T>, 1>(regions[idx], rect, extra_dim_in, offset + idx - 1);
      const size_t volume = rect.volume();
      const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      legate_argmin_radix_1d<T><<<blocks, THREADS_PER_BLOCK>>>(out, args, num_inputs, rect.lo, volume);
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<Argval<T>, 2> out =
          (extra_dim_out >= 0)
              ? derez.unpack_accessor_WO<Argval<T>, 2>(regions[0], rect, extra_dim_out, task->index_point[extra_dim_out])
              : derez.unpack_accessor_WO<Argval<T>, 2>(regions[0], rect);
      ArgminRadixArgs<T, 2> args;
      unsigned              num_inputs = 0;
      for (unsigned idx = 1; idx < task->regions.size(); idx++)
        if (task->regions[idx].region.exists())
          args.in[num_inputs++] = derez.unpack_accessor_RO<Argval<T>, 2>(regions[idx], rect, extra_dim_in, offset + idx - 1);
      const size_t  volume = rect.volume();
      const size_t  blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t pitch  = rect.hi[1] - rect.lo[1] + 1;
      legate_argmin_radix_2d<T><<<blocks, THREADS_PER_BLOCK>>>(out, args, num_inputs, rect.lo, Point<1>(pitch), volume);
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<Argval<T>, 3> out =
          (extra_dim_out >= 0)
              ? derez.unpack_accessor_WO<Argval<T>, 3>(regions[0], rect, extra_dim_out, task->index_point[extra_dim_out])
              : derez.unpack_accessor_WO<Argval<T>, 3>(regions[0], rect);
      ArgminRadixArgs<T, 3> args;
      unsigned              num_inputs = 0;
      for (unsigned idx = 1; idx < task->regions.size(); idx++)
        args.in[num_inputs++] = derez.unpack_accessor_RO<Argval<T>, 3>(regions[idx], rect, extra_dim_in, offset + idx - 1);
      const size_t  volume   = rect.volume();
      const size_t  blocks   = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t diffy    = rect.hi[1] - rect.lo[1] + 1;
      const coord_t diffz    = rect.hi[2] - rect.lo[2] + 1;
      const coord_t pitch[2] = {diffy * diffz, diffz};
      legate_argmin_radix_3d<T><<<blocks, THREADS_PER_BLOCK>>>(out, args, num_inputs, rect.lo, Point<2>(pitch), volume);
      break;
    }
    default:
      assert(false);
  }
}

INSTANTIATE_TASK_VARIANT(ArgminRadixTask, gpu_variant)

}    // namespace numpy
}    // namespace legate
