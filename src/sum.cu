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
#include "proj.h"
#include "sum.h"

using namespace Legion;

namespace legate {
namespace numpy {

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_sum_2d(const AccessorRW<T, 2> inout,
                const AccessorRO<T, 2> in,
                const Rect<2> bounds,
                const T identity,
                const int axis)
{
  coord_t y = bounds.lo[1] + blockIdx.x * blockDim.x + threadIdx.x;
  coord_t x = bounds.lo[0] + (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y + threadIdx.y;
  const Point<2> p(x, y);
  if (!bounds.contains(p)) return;
  T value = identity;
  // In most cases there will be one thread per array element, and all threads whose coordinates
  // differ only on the collapsed axis will atomically reduce into the same output location.
  // One optimization we do is to avoid spawning more threadblocks than will fit on the device,
  // by under-subscribing on the collapsed dimension. We handle this with a grid-striding loop.
  if (axis == 0) {
    while (x <= bounds.hi[0]) {
      SumReduction<T>::template fold<true /*exclusive*/>(value, in[x][y]);
      // We piggyback on the z dimension if we go over the CUDA launch bounds limits.
      x += gridDim.z * gridDim.y * blockDim.y;
    }
  } else {
    while (y <= bounds.hi[1]) {
      SumReduction<T>::template fold<true /*exclusive*/>(value, in[x][y]);
      y += gridDim.x * blockDim.x;
    }
#if __CUDA_ARCH__ >= 700
    __shared__ T trampoline[THREADS_PER_BLOCK];
    // Check for the case where all the threads in the same warp have
    // the same x value in which case they're all going to conflict
    // so instead we do a warp-level reduction so just one thread ends
    // up doing the full atomic
    const int same_mask = __match_any_sync(0xffffffff, threadIdx.y);
    int laneid;
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
            } else  // Make sure we don't do this test again
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

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_sum_3d(const AccessorRW<T, 3> inout,
                const AccessorRO<T, 3> in,
                const Rect<3> bounds,
                const T identity,
                const int axis)
{
  coord_t z = bounds.lo[2] + blockIdx.x * blockDim.x + threadIdx.x;
  coord_t y = bounds.lo[1] + blockIdx.y * blockDim.y + threadIdx.y;
  coord_t x = bounds.lo[0] + blockIdx.z * blockDim.z + threadIdx.z;
  const Point<3> p(x, y, z);
  if (!bounds.contains(p)) return;
  T value = identity;
  if (axis == 0) {
    while (x <= bounds.hi[0]) {
      SumReduction<T>::template fold<true /*exclusive*/>(value, in[x][y][z]);
      x += gridDim.z * blockDim.z;
    }
  } else if (axis == 1) {
    while (y <= bounds.hi[1]) {
      SumReduction<T>::template fold<true /*exclusive*/>(value, in[x][y][z]);
      y += gridDim.y * blockDim.y;
    }
  } else {
    while (z <= bounds.hi[2]) {
      SumReduction<T>::template fold<true /*exclusive*/>(value, in[x][y][z]);
      z += gridDim.x * blockDim.x;
    }
#if __CUDA_ARCH__ >= 700
    __shared__ T trampoline[THREADS_PER_BLOCK];
    // Check for the case where all the threads in the same warp have
    // the same x value in which case they're all going to conflict
    // so instead we do a warp-level reduction so just one thread ends
    // up doing the full atomic
    const int same_mask = __match_any_sync(0xffffffff, threadIdx.z * blockDim.y + threadIdx.y);
    int laneid;
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
            } else  // Make sure we don't do this test again
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

template <typename T>
/*static*/ void SumTask<T>::gpu_variant(const Task* task,
                                        const std::vector<PhysicalRegion>& regions,
                                        Context ctx,
                                        Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int axis         = derez.unpack_dimension();
  const int collapse_dim = derez.unpack_dimension();
  const int init_dim     = derez.unpack_dimension();
  const T initial_value =
    (task->futures.size() == 1) ? task->futures[0].get_result<T>() : SumReduction<T>::identity;
  switch (init_dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<T, 1> out =
        (collapse_dim >= 0) ? derez.unpack_accessor_WO<T, 1>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_WO<T, 1>(regions[0], rect);
      const size_t volume = rect.volume();
      const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      legate_fill_1d<T><<<blocks, THREADS_PER_BLOCK>>>(out, initial_value, rect.lo, volume);
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<T, 2> out =
        (collapse_dim >= 0) ? derez.unpack_accessor_WO<T, 2>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_WO<T, 2>(regions[0], rect);
      const size_t volume = rect.volume();
      const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t pitch = rect.hi[1] - rect.lo[1] + 1;
      legate_fill_2d<T>
        <<<blocks, THREADS_PER_BLOCK>>>(out, initial_value, rect.lo, Point<1>(pitch), volume);
      break;
    }
    default: assert(false);  // shouldn't see any other cases
  }
  const int dim = derez.unpack_dimension();
  switch (dim) {
    // Should never get the case of 1 as this would just be a copy since
    // reducing our only dimension should have called SumReducTask
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorRW<T, 2> inout =
        (collapse_dim >= 0) ? derez.unpack_accessor_RW<T, 2, 1>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_RW<T, 2>(regions[0], rect);
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      // Figure out how many blocks and threads we need
      dim3 threads(1, 1, 1);
      dim3 blocks(1, 1, 1);
      raster_2d_reduction(blocks, threads, rect, axis, (const void*)legate_sum_2d<T>);
      legate_sum_2d<T><<<blocks, threads>>>(inout, in, rect, SumReduction<T>::identity, axis);
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) return;
      const AccessorRW<T, 3> inout =
        (collapse_dim >= 0) ? derez.unpack_accessor_RW<T, 3, 2>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_RW<T, 3>(regions[0], rect);
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      // Figure out how many blocks and threads we need
      dim3 threads(1, 1, 1);
      dim3 blocks(1, 1, 1);
      raster_3d_reduction(blocks, threads, rect, axis, (const void*)legate_sum_3d<T>);
      legate_sum_3d<T><<<blocks, threads>>>(inout, in, rect, SumReduction<T>::identity, axis);
      break;
    }
    default: assert(false);
  }
}

INSTANTIATE_TASK_VARIANT(SumTask, gpu_variant)

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_sum_reduce_1d(const DeferredBuffer<T, 1> buffer,
                       const AccessorRO<T, 1> in,
                       const Point<1> origin,
                       const size_t max,
                       const T identity)
{
  T value             = identity;
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < max) {
    const coord_t x = origin[0] + offset;
    value           = in[x];
  }
  fold_output(buffer, value, SumReduction<T>{});
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_sum_reduce_2d(const DeferredBuffer<T, 1> buffer,
                       const AccessorRO<T, 2> in,
                       const Point<2> origin,
                       const Point<1> pitch,
                       const size_t max,
                       const T identity)
{
  T value             = identity;
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < max) {
    const coord_t x = origin[0] + offset / pitch[0];
    const coord_t y = origin[1] + offset % pitch[0];
    value           = in[x][y];
  }
  fold_output(buffer, value, SumReduction<T>{});
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_sum_reduce_3d(const DeferredBuffer<T, 1> buffer,
                       const AccessorRO<T, 3> in,
                       const Point<3> origin,
                       const Point<2> pitch,
                       const size_t max,
                       const T identity)
{
  T value             = identity;
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < max) {
    const coord_t x = origin[0] + offset / pitch[0];
    const coord_t y = origin[1] + (offset % pitch[0]) / pitch[1];
    const coord_t z = origin[2] + (offset % pitch[0]) % pitch[1];
    value           = in[x][y][z];
  }
  fold_output(buffer, value, SumReduction<T>{});
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) legate_buffer_sum_reduce(
  const DeferredBuffer<T, 1> in, const DeferredBuffer<T, 1> out, const size_t max, const T identity)
{
  T value             = identity;
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < max) value = in.read(offset);
  fold_output(out, value, SumReduction<T>{});
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_final_sum_reduce(const DeferredBuffer<T, 1> in,
                          const DeferredReduction<SumReduction<T>> out,
                          const size_t max,
                          const T identity)
{
  T value             = identity;
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < max) value = in.read(offset);
  reduce_output(out, value);
}

template <typename T>
/*static*/ DeferredReduction<SumReduction<T>> SumReducTask<T>::gpu_variant(
  const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int dim = derez.unpack_dimension();
  DeferredBuffer<T, 1> bufferA;
  size_t volume = 0, blocks = 0;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
      volume                    = rect.volume();
      blocks                    = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      Rect<1> bounds(Point<1>(0), Point<1>(blocks - 1));
      bufferA = DeferredBuffer<T, 1>(Memory::GPU_FB_MEM, Domain(bounds));
      legate_sum_reduce_1d<T>
        <<<blocks, THREADS_PER_BLOCK>>>(bufferA, in, rect.lo, volume, SumReduction<T>::identity);
      volume = blocks;
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[0], rect);
      volume                    = rect.volume();
      blocks                    = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      Rect<1> bounds(Point<1>(0), Point<1>(blocks - 1));
      bufferA             = DeferredBuffer<T, 1>(Memory::GPU_FB_MEM, Domain(bounds));
      const coord_t pitch = rect.hi[1] - rect.lo[1] + 1;
      legate_sum_reduce_2d<T><<<blocks, THREADS_PER_BLOCK>>>(
        bufferA, in, rect.lo, Point<1>(pitch), volume, SumReduction<T>::identity);
      volume = blocks;
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[0], rect);
      volume                    = rect.volume();
      blocks                    = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      Rect<1> bounds(Point<1>(0), Point<1>(blocks - 1));
      bufferA                = DeferredBuffer<T, 1>(Memory::GPU_FB_MEM, Domain(bounds));
      const coord_t diffy    = rect.hi[1] - rect.lo[1] + 1;
      const coord_t diffz    = rect.hi[2] - rect.lo[2] + 1;
      const coord_t pitch[2] = {diffy * diffz, diffz};
      legate_sum_reduce_3d<T><<<blocks, THREADS_PER_BLOCK>>>(
        bufferA, in, rect.lo, Point<2>(pitch), volume, SumReduction<T>::identity);
      volume = blocks;
      break;
    }
    default: assert(false);
  }
  // Continue reducing buffers until we get down to one small enough that
  // it can be handled by a single CTA and then we can do the final launch
  DeferredBuffer<T, 1> last = bufferA;
  if (volume > THREADS_PER_BLOCK) {
    DeferredBuffer<T, 1> bufferB;
    bool b_initialized = false;
    bool forward       = true;
    while (volume > THREADS_PER_BLOCK) {
      blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      if (!b_initialized) {
        Rect<1> bounds = Rect<1>(Point<1>(0), Point<1>(blocks - 1));
        bufferB        = DeferredBuffer<T, 1>(Memory::GPU_FB_MEM, Domain(bounds));
        b_initialized  = true;
      }
      if (forward) {
        legate_buffer_sum_reduce<T>
          <<<blocks, THREADS_PER_BLOCK>>>(bufferA, bufferB, volume, SumReduction<T>::identity);
        forward = false;
      } else {
        legate_buffer_sum_reduce<T>
          <<<blocks, THREADS_PER_BLOCK>>>(bufferB, bufferA, volume, SumReduction<T>::identity);
        forward = true;
      }
      volume = blocks;
    }
    if (!forward) last = bufferB;
  }
  DeferredReduction<SumReduction<T>> result;
  // One last kernel launch to do the final reduction to a single value
  if (volume > 0)
    legate_final_sum_reduce<T>
      <<<1, THREADS_PER_BLOCK>>>(last, result, volume, SumReduction<T>::identity);
  return result;
}

INSTANTIATE_DEFERRED_REDUCTION_TASK_VARIANT(SumReducTask, SumReduction, gpu_variant)

template <typename T, int DIM>
struct SumRadixArgs {
  AccessorRO<T, DIM> in[MAX_REDUCTION_RADIX];
};

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_sum_radix_1d(const AccessorWO<T, 1> out,
                      const SumRadixArgs<T, 1> args,
                      const size_t argmax,
                      const Point<1> origin,
                      const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset;
  T val           = args.in[0][x];
  for (unsigned idx = 1; idx < argmax; idx++)
    SumReduction<T>::template fold<true /*exclusive*/>(val, args.in[idx][x]);
  out[x] = val;
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_sum_radix_2d(const AccessorWO<T, 2> out,
                      const SumRadixArgs<T, 2> args,
                      const size_t argmax,
                      const Point<2> origin,
                      const Point<1> pitch,
                      const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + offset % pitch[0];
  T val           = args.in[0][x][y];
  for (unsigned idx = 1; idx < argmax; idx++)
    SumReduction<T>::template fold<true /*exclusive*/>(val, args.in[idx][x][y]);
  out[x][y] = val;
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_sum_radix_3d(const AccessorWO<T, 3> out,
                      const SumRadixArgs<T, 3> args,
                      const size_t argmax,
                      const Point<3> origin,
                      const Point<2> pitch,
                      const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z = origin[2] + (offset % pitch[0]) % pitch[1];
  T val           = args.in[0][x][y][z];
  for (unsigned idx = 1; idx < argmax; idx++)
    SumReduction<T>::template fold<true /*exclusive*/>(val, args.in[idx][x][y][z]);
  out[x][y][z] = val;
}

template <typename T>
/*static*/ void SumRadixTask<T>::gpu_variant(const Task* task,
                                             const std::vector<PhysicalRegion>& regions,
                                             Context ctx,
                                             Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  assert(task->regions.size() <= MAX_REDUCTION_RADIX);
  const int radix         = derez.unpack_dimension();
  const int extra_dim_out = derez.unpack_dimension();
  const int extra_dim_in  = derez.unpack_dimension();
  const int dim           = derez.unpack_dimension();
  const coord_t offset    = (extra_dim_in >= 0) ? task->index_point[extra_dim_in] * radix : 0;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 1> out =
        (extra_dim_out >= 0) ? derez.unpack_accessor_WO<T, 1>(
                                 regions[0], rect, extra_dim_out, task->index_point[extra_dim_out])
                             : derez.unpack_accessor_WO<T, 1>(regions[0], rect);
      SumRadixArgs<T, 1> args;
      unsigned num_inputs = 0;
      for (unsigned idx = 1; idx < task->regions.size(); idx++)
        if (task->regions[idx].region.exists())
          args.in[num_inputs++] =
            derez.unpack_accessor_RO<T, 1>(regions[idx], rect, extra_dim_in, offset + idx - 1);
      const size_t volume = rect.volume();
      const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      legate_sum_radix_1d<T><<<blocks, THREADS_PER_BLOCK>>>(out, args, num_inputs, rect.lo, volume);
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 2> out =
        (extra_dim_out >= 0) ? derez.unpack_accessor_WO<T, 2>(
                                 regions[0], rect, extra_dim_out, task->index_point[extra_dim_out])
                             : derez.unpack_accessor_WO<T, 2>(regions[0], rect);
      SumRadixArgs<T, 2> args;
      unsigned num_inputs = 0;
      for (unsigned idx = 1; idx < task->regions.size(); idx++)
        if (task->regions[idx].region.exists())
          args.in[num_inputs++] =
            derez.unpack_accessor_RO<T, 2>(regions[idx], rect, extra_dim_in, offset + idx - 1);
      const size_t volume = rect.volume();
      const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t pitch = rect.hi[1] - rect.lo[1] + 1;
      legate_sum_radix_2d<T>
        <<<blocks, THREADS_PER_BLOCK>>>(out, args, num_inputs, rect.lo, Point<1>(pitch), volume);
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 3> out =
        (extra_dim_out >= 0) ? derez.unpack_accessor_WO<T, 3>(
                                 regions[0], rect, extra_dim_out, task->index_point[extra_dim_out])
                             : derez.unpack_accessor_WO<T, 3>(regions[0], rect);
      SumRadixArgs<T, 3> args;
      unsigned num_inputs = 0;
      for (unsigned idx = 1; idx < task->regions.size(); idx++)
        args.in[num_inputs++] =
          derez.unpack_accessor_RO<T, 3>(regions[idx], rect, extra_dim_in, offset + idx - 1);
      const size_t volume    = rect.volume();
      const size_t blocks    = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t diffy    = rect.hi[1] - rect.lo[1] + 1;
      const coord_t diffz    = rect.hi[2] - rect.lo[2] + 1;
      const coord_t pitch[2] = {diffy * diffz, diffz};
      legate_sum_radix_3d<T>
        <<<blocks, THREADS_PER_BLOCK>>>(out, args, num_inputs, rect.lo, Point<2>(pitch), volume);
      break;
    }
    default: assert(false);
  }
}

INSTANTIATE_TASK_VARIANT(SumRadixTask, gpu_variant)

}  // namespace numpy
}  // namespace legate
