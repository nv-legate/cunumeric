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
#include "nonzero.h"
#include "proj.h"
#include <cstdio>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>
#include <utility>

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_nonzero_2d(const AccessorRW<uint64_t, 2> inout, const AccessorRO<T, 2> in, const Rect<2> bounds, const uint64_t identity,
                      const int axis) {
  coord_t        y = bounds.lo[1] + blockIdx.x * blockDim.x + threadIdx.x;
  coord_t        x = bounds.lo[0] + (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y + threadIdx.y;
  const Point<2> p(x, y);
  if (!bounds.contains(p)) return;
  uint64_t value = identity;
  if (axis == 0) {
    while (x <= bounds.hi[0]) {
      SumReduction<uint64_t>::template fold<true /*exclusive*/>(value, static_cast<uint64_t>(in[x][y] != (T)0));
      x += gridDim.z * gridDim.y * blockDim.y;
    }
  } else {
    while (y <= bounds.hi[1]) {
      SumReduction<uint64_t>::template fold<true /*exclusive*/>(value, static_cast<uint64_t>(in[x][y] != (T)0));
      y += gridDim.x * blockDim.x;
    }
#if __CUDA_ARCH__ >= 700
    __shared__ uint64_t trampoline[THREADS_PER_BLOCK];
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
            SumReduction<uint64_t>::template fold<true /*exclusive*/>(value, trampoline[index]);
          }
        }
    }
#endif
  }
  if (value != identity) SumReduction<uint64_t>::template fold<false /*exclusive*/>(inout[p], value);
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_nonzero_3d(const AccessorRW<uint64_t, 3> inout, const AccessorRO<T, 3> in, const Rect<3> bounds, const uint64_t identity,
                      const int axis) {
  coord_t        z = bounds.lo[2] + blockIdx.x * blockDim.x + threadIdx.x;
  coord_t        y = bounds.lo[1] + blockIdx.y * blockDim.y + threadIdx.y;
  coord_t        x = bounds.lo[0] + blockIdx.z * blockDim.z + threadIdx.z;
  const Point<3> p(x, y, z);
  if (!bounds.contains(p)) return;
  uint64_t value = identity;
  if (axis == 0) {
    while (x <= bounds.hi[0]) {
      SumReduction<uint64_t>::template fold<true /*exclusive*/>(value, static_cast<uint64_t>(in[x][y][z] != (T)0));
      x += gridDim.z * blockDim.z;
    }
  } else if (axis == 1) {
    while (y <= bounds.hi[1]) {
      SumReduction<uint64_t>::template fold<true /*exclusive*/>(value, static_cast<uint64_t>(in[x][y][z] != (T)0));
      y += gridDim.y * blockDim.y;
    }
  } else {
    while (z <= bounds.hi[2]) {
      SumReduction<uint64_t>::template fold<true /*exclusive*/>(value, static_cast<uint64_t>(in[x][y][z] != (T)0));
      z += gridDim.x * blockDim.x;
    }
#if __CUDA_ARCH__ >= 700
    __shared__ uint64_t trampoline[THREADS_PER_BLOCK];
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
            SumReduction<uint64_t>::template fold<true /*exclusive*/>(value, trampoline[index]);
          }
        }
    }
#endif
  }
  if (value != identity) SumReduction<uint64_t>::template fold<false /*exclusive*/>(inout[p], value);
}

template<typename T>
/*static*/ void CountNonzeroTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                 Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          axis         = derez.unpack_dimension();
  const int          collapse_dim = derez.unpack_dimension();
  const int          init_dim     = derez.unpack_dimension();
  switch (init_dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<uint64_t, 1> out =
          (collapse_dim >= 0)
              ? derez.unpack_accessor_WO<uint64_t, 1>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
              : derez.unpack_accessor_WO<uint64_t, 1>(regions[0], rect);
      const coord_t volume = rect.volume();
      const coord_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      legate_fill_1d<uint64_t><<<blocks, THREADS_PER_BLOCK>>>(out, SumReduction<uint64_t>::identity, rect.lo, volume);
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<uint64_t, 2> out =
          (collapse_dim >= 0)
              ? derez.unpack_accessor_WO<uint64_t, 2>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
              : derez.unpack_accessor_WO<uint64_t, 2>(regions[0], rect);
      const coord_t volume = rect.volume();
      const coord_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t pitch  = rect.hi[1] - rect.lo[1] + 1;
      legate_fill_2d<uint64_t>
          <<<blocks, THREADS_PER_BLOCK>>>(out, SumReduction<uint64_t>::identity, rect.lo, Point<1>(pitch), volume);
      break;
    }
    default:
      assert(false);    // shouldn't see any other cases
  }
  const int dim = derez.unpack_dimension();
  switch (dim) {
    // Should never get the case of 1 as this would just be a copy since
    // reducing our only dimension should have called SumReducTask
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRW<uint64_t, 2> inout =
          (collapse_dim >= 0)
              ? derez.unpack_accessor_RW<uint64_t, 2, 1>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
              : derez.unpack_accessor_RW<uint64_t, 2>(regions[0], rect);
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      // Figure out how many blocks and threads we need
      dim3 threads(1, 1, 1);
      dim3 blocks(1, 1, 1);
      raster_2d_reduction(blocks, threads, rect, axis, (const void*)legate_nonzero_2d<T>);
      legate_nonzero_2d<T><<<blocks, threads>>>(inout, in, rect, SumReduction<uint64_t>::identity, axis);
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRW<uint64_t, 3> inout =
          (collapse_dim >= 0)
              ? derez.unpack_accessor_RW<uint64_t, 3, 2>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
              : derez.unpack_accessor_RW<uint64_t, 3>(regions[0], rect);
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      // Figure out how many blocks and threads we need
      dim3 threads(1, 1, 1);
      dim3 blocks(1, 1, 1);
      raster_3d_reduction(blocks, threads, rect, axis, (const void*)legate_nonzero_3d<T>);
      legate_nonzero_3d<T><<<blocks, threads>>>(inout, in, rect, SumReduction<uint64_t>::identity, axis);
      break;
    }
    default:
      assert(false);
  }
}

INSTANTIATE_TASK_VARIANT(CountNonzeroTask, gpu_variant)

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_nonzero_reduce_1d(const DeferredBuffer<uint64_t, 1> buffer, const AccessorRO<T, 1> in, const Point<1> origin,
                             const size_t max, const uint64_t identity) {
  uint64_t     value  = identity;
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < max) {
    const coord_t x = origin[0] + offset;
    value += static_cast<uint64_t>(in[x] != (T)0);
  }
  fold_output(buffer, value, SumReduction<uint64_t>{});
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_nonzero_reduce_2d(const DeferredBuffer<uint64_t, 1> buffer, const AccessorRO<T, 2> in, const Point<2> origin,
                             const Point<1> pitch, const size_t max, const uint64_t identity) {
  uint64_t     value  = identity;
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < max) {
    const coord_t x = origin[0] + offset / pitch[0];
    const coord_t y = origin[1] + offset % pitch[0];
    value += static_cast<uint64_t>(in[x][y] != (T)0);
  }
  fold_output(buffer, value, SumReduction<uint64_t>{});
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_nonzero_reduce_3d(const DeferredBuffer<uint64_t, 1> buffer, const AccessorRO<T, 3> in, const Point<3> origin,
                             const Point<2> pitch, const size_t max, const uint64_t identity) {
  uint64_t     value  = identity;
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < max) {
    const coord_t x = origin[0] + offset / pitch[0];
    const coord_t y = origin[1] + (offset % pitch[0]) / pitch[1];
    const coord_t z = origin[2] + (offset % pitch[0]) % pitch[1];
    value += static_cast<uint64_t>(in[x][y][z] != (T)0);
  }
  fold_output(buffer, value, SumReduction<uint64_t>{});
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_buffer_nonzero_reduce(const DeferredBuffer<T, 1> in, const DeferredBuffer<uint64_t, 1> out, const size_t max,
                                 const uint64_t identity) {
  uint64_t     value  = identity;
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < max) value += in.read(offset);
  fold_output(out, value, SumReduction<uint64_t>{});
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_final_nonzero_reduce(const DeferredBuffer<T, 1> in, const DeferredReduction<SumReduction<uint64_t>> out,
                                const size_t max, const uint64_t identity) {
  uint64_t     value  = identity;
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < max) value += in.read(offset);
  reduce_output(out, value);
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_final_nonzero_reduce(const DeferredBuffer<T, 1> in, T& out, const size_t max, const uint64_t identity) {
  uint64_t     value  = identity;
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < max) value += in.read(offset);
  fold_output(in, value, SumReduction<uint64_t>{});
  out = in.read(blockIdx.x);
}

namespace detail {
template<typename T>
/*static*/ std::tuple<coord_t, DeferredBuffer<uint64_t, 1>>
    count_nonzero_reduc_task_gpu_helper(const Task* task, const std::vector<PhysicalRegion>& regions, LegateDeserializer& derez) {
  const int                   dim = derez.unpack_dimension();
  DeferredBuffer<uint64_t, 1> bufferA;
  coord_t                     volume = 0, blocks = 0;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
      volume                    = rect.volume();
      blocks                    = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      Rect<1> bounds(Point<1>(0), Point<1>(blocks - 1));
      bufferA = DeferredBuffer<uint64_t, 1>(Memory::GPU_FB_MEM, Domain(bounds));
      legate_nonzero_reduce_1d<T><<<blocks, THREADS_PER_BLOCK>>>(bufferA, in, rect.lo, volume, SumReduction<uint64_t>::identity);
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
      bufferA             = DeferredBuffer<uint64_t, 1>(Memory::GPU_FB_MEM, Domain(bounds));
      const coord_t pitch = rect.hi[1] - rect.lo[1] + 1;
      legate_nonzero_reduce_2d<T>
          <<<blocks, THREADS_PER_BLOCK>>>(bufferA, in, rect.lo, Point<1>(pitch), volume, SumReduction<uint64_t>::identity);
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
      bufferA                = DeferredBuffer<uint64_t, 1>(Memory::GPU_FB_MEM, Domain(bounds));
      const coord_t diffy    = rect.hi[1] - rect.lo[1] + 1;
      const coord_t diffz    = rect.hi[2] - rect.lo[2] + 1;
      const coord_t pitch[2] = {diffy * diffz, diffz};
      legate_nonzero_reduce_3d<T>
          <<<blocks, THREADS_PER_BLOCK>>>(bufferA, in, rect.lo, Point<2>(pitch), volume, SumReduction<uint64_t>::identity);
      volume = blocks;
      break;
    }
    default:
      assert(false);
  }
  // Continue reducing buffers until we get down to one small enough that
  // it can be handled by a single CTA and then we can do the final launch
  DeferredBuffer<uint64_t, 1> last = bufferA;
  if (volume > THREADS_PER_BLOCK) {
    DeferredBuffer<uint64_t, 1> bufferB;
    bool                        b_initialized = false;
    bool                        forward       = true;
    while (volume > THREADS_PER_BLOCK) {
      blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      if (!b_initialized) {
        Rect<1> bounds = Rect<1>(Point<1>(0), Point<1>(blocks - 1));
        bufferB        = DeferredBuffer<uint64_t, 1>(Memory::GPU_FB_MEM, Domain(bounds));
        b_initialized  = true;
      }
      if (forward) {
        legate_buffer_nonzero_reduce<uint64_t>
            <<<blocks, THREADS_PER_BLOCK>>>(bufferA, bufferB, volume, SumReduction<uint64_t>::identity);
        forward = false;
      } else {
        legate_buffer_nonzero_reduce<uint64_t>
            <<<blocks, THREADS_PER_BLOCK>>>(bufferB, bufferA, volume, SumReduction<uint64_t>::identity);
        forward = true;
      }
      volume = blocks;
    }
    if (!forward) last = bufferB;
  }
  return {volume, last};
}
}    // namespace detail

template<typename T>
/*static*/ DeferredReduction<SumReduction<uint64_t>>
    CountNonzeroReducTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                          Runtime* runtime) {
  LegateDeserializer          derez(task->args, task->arglen);
  coord_t                     volume{};
  DeferredBuffer<uint64_t, 1> buffer{};
  std::tie(volume, buffer) = detail::count_nonzero_reduc_task_gpu_helper<T>(task, regions, derez);
  DeferredReduction<SumReduction<uint64_t>> result;
  // One last kernel launch to do the final reduction to a single value
  if (volume > 0)
    legate_final_nonzero_reduce<uint64_t><<<1, THREADS_PER_BLOCK>>>(buffer, result, volume, SumReduction<uint64_t>::identity);
  return result;
}

template<typename T>
/*static*/ void CountNonzeroReducWriteTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions,
                                                           Context ctx, Runtime* runtime) {
  LegateDeserializer          derez(task->args, task->arglen);
  coord_t                     volume{};
  DeferredBuffer<uint64_t, 1> buffer{};
  std::tie(volume, buffer) = detail::count_nonzero_reduc_task_gpu_helper<T>(task, regions, derez);
  const int dim            = derez.unpack_dimension();
  if (dim == 1) {
    const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
    assert(rect.volume() == 1);
    const AccessorWO<uint64_t, 1> out = derez.unpack_accessor_WO<uint64_t, 1>(regions[1], rect);
    if (volume > 0)
      legate_final_nonzero_reduce<uint64_t>
          <<<1, THREADS_PER_BLOCK>>>(buffer, out[rect.lo], volume, SumReduction<uint64_t>::identity);
  } else if (dim == 2) {
    const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
    assert(rect.volume() == 1);
    const AccessorWO<uint64_t, 2> out = derez.unpack_accessor_WO<uint64_t, 2>(regions[1], rect);
    if (volume > 0)
      legate_final_nonzero_reduce<uint64_t>
          <<<1, THREADS_PER_BLOCK>>>(buffer, out[rect.lo], volume, SumReduction<uint64_t>::identity);
  } else if (dim == 3) {
    const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
    assert(rect.volume() == 1);
    const AccessorWO<uint64_t, 3> out = derez.unpack_accessor_WO<uint64_t, 3>(regions[1], rect);
    if (volume > 0)
      legate_final_nonzero_reduce<uint64_t>
          <<<1, THREADS_PER_BLOCK>>>(buffer, out[rect.lo], volume, SumReduction<uint64_t>::identity);
  }
}

INSTANTIATE_DEFERRED_REDUCTION_ARG_RETURN_TASK_VARIANT(CountNonzeroReducTask, SumReduction, gpu_variant)
INSTANTIATE_TASK_VARIANT(CountNonzeroReducWriteTask, gpu_variant)

namespace detail {

template<typename Tuple, size_t... Is>
__device__ __host__ auto make_thrust_tuple_impl(Tuple&& t, std::index_sequence<Is...>) {
  return thrust::make_tuple(std::get<Is>(std::forward<Tuple>(t))...);
}

template<typename Tuple>
__device__ __host__ auto make_thrust_tuple(Tuple&& t) {
  using Indices = std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>;
  return make_thrust_tuple_impl(std::forward<Tuple>(t), Indices{});
}

template<size_t dims, typename Rect>
__device__ __host__ size_t volume(const Rect rect, size_t level) {
  size_t ret = 1;
  for (auto i = level; i < dims; ++i) {
    ret *= rect.hi[i] - rect.lo[i] + 1;
  }
  return ret;
}

template<size_t dims, size_t level>
struct helper {
  template<typename Accessor, typename Rect>
  __device__ __host__ decltype(auto) operator()(Accessor accessor, const Rect rect, const coord_t position) const {
    const size_t  vol    = volume<dims>(rect, level);
    const size_t  offset = position / vol;
    const coord_t index  = rect.lo[level - 1] + offset;
    // cudaDeviceSynchronize();
    return helper<dims, level + 1>{}(accessor[index], rect, position - offset * vol);
  }
};

template<size_t dims>
struct helper<dims, dims> {
  template<typename Accessor, typename Rect>
  __device__ __host__ decltype(auto) operator()(Accessor accessor, const Rect rect, const coord_t position) const {
    // cudaDeviceSynchronize();
    return accessor[static_cast<coord_t>(rect.lo[dims - 1] + position % (rect.hi[dims - 1] + 1))];
  }
};

template<size_t dims>
struct accessor_helper {
  template<typename Accessor, typename Rect>
  __device__ __host__ decltype(auto) operator()(Accessor accessor, const Rect rect, const coord_t position) const {
    // cudaDeviceSynchronize();
    return helper<dims, 1>{}(accessor, rect, position);
  }
};

template<size_t dims, size_t level>
struct ihelper {
  template<typename Rect>
  __device__ __host__ decltype(auto) operator()(const Rect rect, const coord_t position) const {
    const size_t  vol    = volume<dims>(rect, level);
    const size_t  offset = position / vol;
    const coord_t index  = rect.lo[level - 1] + offset;
    // cudaDeviceSynchronize();
    return std::tuple_cat(std::make_tuple(index), ihelper<dims, level + 1>{}(rect, position - offset * vol));
  }
};

template<size_t dims>
struct ihelper<dims, dims> {
  template<typename Rect>
  __device__ __host__ decltype(auto) operator()(const Rect rect, const coord_t position) const {
    // cudaDeviceSynchronize();
    return std::make_tuple(static_cast<coord_t>(rect.lo[dims - 1] + position % (rect.hi[dims - 1] + 1)));
  }
};

template<typename Rect, size_t dims>
struct index_helper {
  __device__ __host__ decltype(auto) operator()(const Rect rect, const coord_t position) const {
    // cudaDeviceSynchronize();
    return make_thrust_tuple(ihelper<dims, 1>{}(rect, position));
  }
};

// template<typename Accessor, typename Rect>
// struct accessor_helper<Accessor, Rect, 1> {
//   __device__ __host__ decltype(auto) operator()(const Accessor& accessor, const Rect& rect, const coord_t position) const {
//     printf("position: %llu, rect.lo[0]: %llu, rect.hi[0]: %llu\n", position, rect.lo[0], rect.hi[0]);
//     printf("Accessing at %llu \n", rect.lo[0] + position);
//     return accessor[rect.lo[0] + position];
//   }

//   // __device__ __host__ decltype(auto) operator()(Accessor& accessor, Rect& rect, coord_t position) const {
//   //   printf("position: %llu, rect.lo[0]: %llu, rect.hi[0]: %llu\n", position, rect.lo[0], rect.hi[0]);
//   //   printf("Accessing at %llu \n", rect.lo[0] + position);
//   //   return accessor[rect.lo[0] + position];
//   // }
// };

// template<typename Accessor, typename Rect>
// struct accessor_helper<Accessor, Rect, 2> {
//   __device__ __host__ decltype(auto) operator()(const Accessor& accessor, const Rect& rect, coord_t position) const {
//     printf("position: %llu, rect.lo[0]: %llu, rect.hi[0]: %llu, rect.lo[1]: %llu, rect.hi[1]: %llu\n", position, rect.lo[0],
//            rect.hi[0], rect.lo[1], rect.hi[1]);
//     printf("Accessing at %llu, %llu \n", rect.lo[0] + position / (rect.hi[1] + 1), rect.lo[1] + position % (rect.hi[1] + 1));
//     return accessor[rect.lo[0] + position / (rect.hi[1] + 1)][rect.lo[1] + position % (rect.hi[1] + 1)];
//   }

//   // __device__ __host__ decltype(auto) operator()(Accessor& accessor, Rect& rect, coord_t position) const {
//   //   printf("position: %llu, rect.lo[0]: %llu, rect.hi[0]: %llu, rect.lo[1]: %llu, rect.hi[1]: %llu\n", position, rect.lo[0],
//   //          rect.hi[0], rect.lo[1], rect.hi[1]);
//   //   printf("Accessing at %llu, %llu \n", rect.lo[0] + position / (rect.hi[1] + 1), rect.lo[1] + position % (rect.hi[1] +
//   1));
//   //   return accessor[rect.lo[0] + position / (rect.hi[1] + 1)][rect.lo[1] + position % (rect.hi[1] + 1)];
//   // }
// };
}    // namespace detail

template<typename Rect, size_t dim = Rect::dim, bool row_major = true>
class index_iterator
    : public thrust::iterator_facade<index_iterator<Rect, dim, row_major>,
                                     std::remove_reference_t<decltype(detail::index_helper<Rect, dim>{}(std::declval<Rect>(), 0))>,
                                     thrust::any_system_tag, thrust::random_access_traversal_tag,
                                     decltype(detail::index_helper<Rect, dim>{}(std::declval<Rect>(), 0)), std::ptrdiff_t> {
public:
  index_iterator(const Rect rect, const coord_t position) : rect{rect}, position{position} {}

private:
  friend class thrust::iterator_core_access;

  __device__ __host__ decltype(auto) dereference() const { return detail::index_helper<Rect, dim>{}(rect, position); }

  __device__ __host__ bool equal(const index_iterator& rhs) const { return rhs.position == position; }

  __device__ __host__ void increment() { position++; }

  __device__ __host__ void advance(std::ptrdiff_t diff) { position += diff; }

  __device__ __host__ std::ptrdiff_t distance_to(const index_iterator& rhs) const { return rhs.position - position; }

  coord_t position{0};
  Rect    rect;

  // Need to implement column major
  static_assert(row_major == true, "column major not implemented");
};

template<size_t dim, typename Rect, bool row_major = true>
index_iterator<Rect, dim, row_major> make_index_iterator(const Rect rect, coord_t position) {
  return index_iterator<Rect, dim, row_major>(rect, position);
}

template<typename Accessor, typename Rect, size_t dim = Accessor::dim, typename T = typename Accessor::value_type,
         bool row_major = true>
class accessor_iterator
    : public thrust::iterator_facade<
          accessor_iterator<Accessor, Rect, dim, T, row_major>, T, thrust::any_system_tag, thrust::random_access_traversal_tag,
          decltype(detail::accessor_helper<dim>{}(std::declval<Accessor>(), std::declval<Rect>(), 0)), std::ptrdiff_t> {
public:
  accessor_iterator(const Accessor accessor, const Rect rect) : accessor{accessor}, rect{rect} {}

private:
  friend class thrust::iterator_core_access;

  __device__ __host__ decltype(auto) dereference() const { return detail::accessor_helper<dim>{}(accessor, rect, position); }

  __device__ __host__ bool equal(const accessor_iterator& rhs) const { return rhs.position == position; }

  __device__ __host__ void increment() { position++; }

  __device__ __host__ void advance(std::ptrdiff_t diff) { position += diff; }

  __device__ __host__ std::ptrdiff_t distance_to(const accessor_iterator& rhs) const { return rhs.position - position; }

  Accessor accessor;
  coord_t  position{0};
  Rect     rect;

  // Need to implement column major
  static_assert(row_major == true, "column major not implemented");
};

template<typename Accessor, typename Rect>
accessor_iterator<Accessor, Rect> make_accessor_iterator(const Accessor acc, const Rect rect) {
  return accessor_iterator<Accessor, Rect>(acc, rect);
}

template<typename T>
/*static*/ void NonzeroTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                            Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          in_dim = derez.unpack_dimension();
  assert(in_dim > 0);
  switch (in_dim) {
    case 1: {
      const Rect<1>                 in_rect  = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      const AccessorRO<T, 1>        in       = derez.unpack_accessor_RO<T, 1>(regions[0], in_rect);
      const Rect<2>                 out_rect = regions[1];
      const AccessorRW<uint64_t, 2> out      = derez.unpack_accessor_RW<uint64_t, 2>(regions[1], out_rect);
      auto                          in_iter  = make_accessor_iterator(in, in_rect);
      auto                          out_iter = make_accessor_iterator(out, out_rect);

      thrust::copy_if(thrust::device, thrust::make_counting_iterator(in_rect.lo[0]),
                      thrust::make_counting_iterator(in_rect.hi[0] + 1), in_iter, out_iter,
                      [] __CUDA_HD__(const T& arg) { return arg != T{0}; });

      break;
    }
    case 2: {
      const Rect<2>                 in_rect  = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      const AccessorRO<T, 2>        in       = derez.unpack_accessor_RO<T, 2>(regions[0], in_rect);
      const Rect<2>                 out_rect = regions[1];
      const AccessorRW<uint64_t, 2> out      = derez.unpack_accessor_RW<uint64_t, 2>(regions[1], out_rect);
      auto                          in_iter  = make_accessor_iterator(in, in_rect);
      auto                          out_iter = make_accessor_iterator(out, out_rect);

      thrust::copy_if(thrust::device, make_index_iterator<2>(in_rect, 0), make_index_iterator<2>(in_rect, in_rect.volume()),
                      in_iter, thrust::make_zip_iterator(thrust::make_tuple(out_iter, out_iter + detail::volume<2>(out_rect, 1))),
                      [] __CUDA_HD__(const T& arg) { return arg != T{0}; });

      break;
    }
    case 3: {
      const Rect<3>                 in_rect  = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      const AccessorRO<T, 3>        in       = derez.unpack_accessor_RO<T, 3>(regions[0], in_rect);
      const Rect<2>                 out_rect = regions[1];
      const AccessorRW<uint64_t, 2> out      = derez.unpack_accessor_RW<uint64_t, 2>(regions[1], out_rect);
      auto                          in_iter  = make_accessor_iterator(in, in_rect);
      auto                          out_iter = make_accessor_iterator(out, out_rect);

      thrust::copy_if(thrust::device, make_index_iterator<3>(in_rect, 0), make_index_iterator<3>(in_rect, in_rect.volume()),
                      in_iter,
                      thrust::make_zip_iterator(thrust::make_tuple(out_iter, out_iter + detail::volume<2>(out_rect, 1),
                                                                   out_iter + 2 * detail::volume<2>(out_rect, 1))),
                      [] __CUDA_HD__(const T& arg) { return arg != T{0}; });

      break;
    }
    default:
      assert(false);
  }
}

INSTANTIATE_TASK_VARIANT(NonzeroTask, gpu_variant)

namespace detail {
struct MakeRect {
  __CUDA_HD__ MakeRect(coord_t nonzero_dim) : nonzero_dim{nonzero_dim} {}

  template<typename T>
  __CUDA_HD__ Rect<2> operator()(const T lo, const T hi) {
    coord_t pt1[2] = {0, static_cast<coord_t>(lo)};
    coord_t pt2[2] = {nonzero_dim, static_cast<coord_t>(hi) - 1};
    return Rect<2>{Point<2>(pt1), Point<2>(pt2)};
  }

  coord_t nonzero_dim;
};
}    // namespace detail

template<typename T>
/*static*/ void ConvertRangeToRectTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                       Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const coord_t      nonzero_dim = derez.unpack_32bit_int();
  const int          dim         = derez.unpack_dimension();
  assert(dim == 1);
  const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  assert(!rect.empty());
  const auto                   begin    = rect.lo[0];
  const auto                   end      = rect.hi[0];
  const AccessorRO<T, 1>       in       = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
  const AccessorRW<Rect<2>, 1> out      = derez.unpack_accessor_RW<Rect<2>, 1>(regions[1], rect);
  auto                         in_iter  = make_accessor_iterator(in, rect);
  auto                         out_iter = make_accessor_iterator(out, rect);
  auto const                   size     = rect.volume();

  Rect<1> bounds(Point<1>(0), Point<1>(size + 1));
  auto    buffer = DeferredBuffer<T, 1>(Memory::GPU_FB_MEM, Domain(bounds));
  thrust::uninitialized_fill(thrust::device, out_iter, out_iter + size, Rect<2>{});
  // We just really need to initialize the first value to 0, but uninitialized fill of the whole buffer is just easier to do
  thrust::uninitialized_fill(thrust::device, buffer.ptr(0), buffer.ptr(0) + size + 1, 0);
  thrust::copy(thrust::device, in_iter, in_iter + size, buffer.ptr(0) + 1);
  thrust::transform(thrust::device, buffer.ptr(0), buffer.ptr(0) + size, buffer.ptr(0) + 1, out_iter,
                    detail::MakeRect{nonzero_dim});
}

INSTANTIATE_INT_VARIANT(ConvertRangeToRectTask, gpu_variant)
INSTANTIATE_UINT_VARIANT(ConvertRangeToRectTask, gpu_variant)

}    // namespace numpy
}    // namespace legate
