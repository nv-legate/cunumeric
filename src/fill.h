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

#ifndef __NUMPY_FILL_H__
#define __NUMPY_FILL_H__

#include "point_task.h"

namespace legate {
namespace numpy {

#if defined(LEGATE_USE_CUDA) && defined(__CUDACC__)
template <int DIM, typename Args>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) gpu_fill(const Args args)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= args.volume) return;
  const Legion::Point<DIM> point = args.pitches.unflatten(idx, args.rect.lo);
  args.out[point]                = args.value;
}
#endif

template <typename T>
struct FillTask : PointTask<FillTask<T>> {
  static const int TASK_ID = task_id<NumPyOpCode::NUMPY_FILL, NUMPY_NORMAL_VARIANT_OFFSET, T>;
  static const int REGIONS = 1;

  using result_type = T;

  template <int N>
  struct DeserializedArgs {
    Legion::Rect<N> rect;
    AccessorWO<result_type, N> out;
    Pitches<N - 1> pitches;
    size_t volume;
    result_type value;
    result_type* outptr;
    bool deserialize(LegateDeserializer& derez,
                     const Legion::Task* task,
                     const std::vector<Legion::PhysicalRegion>& regions)
    {
      rect   = NumPyProjectionFunctor::unpack_shape<N>(task, derez);
      out    = derez.unpack_accessor_WO<result_type, N>(regions[0], rect);
      value  = task->futures[0].get_result<result_type>(true /*silence warnings*/);
      volume = pitches.flatten(rect);
#ifndef LEGION_BOUNDS_CHECKS
      // Check to see if this is dense or not
      return out.accessor.is_dense_row_major(rect) && (outptr = out.ptr(rect));
#else
      // No dense execution if we're doing bounds checks
      return false;
#endif
    }
  };

  template <int DIM>
  static void dispatch_cpu(const Legion::Task* task,
                           const std::vector<Legion::PhysicalRegion>& regions,
                           LegateDeserializer& derez)
  {
    DeserializedArgs<DIM> args;
    const bool dense = args.deserialize(derez, task, regions);
    if (args.volume == 0) return;
    if (dense) {
      for (size_t idx = 0; idx < args.volume; ++idx) args.outptr[idx] = args.value;
    } else {
      for (size_t idx = 0; idx < args.volume; ++idx) {
        const Legion::Point<DIM> point = args.pitches.unflatten(idx, args.rect.lo);
        args.out[point]                = args.value;
      }
    }
  }

#ifdef LEGATE_USE_OPENMP
  template <int DIM>
  static void dispatch_omp(const Legion::Task* task,
                           const std::vector<Legion::PhysicalRegion>& regions,
                           LegateDeserializer& derez)
  {
    DeserializedArgs<DIM> args;
    const bool dense = args.deserialize(derez, task, regions);
    if (args.volume == 0) return;
    if (dense) {
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < args.volume; ++idx) { args.outptr[idx] = args.value; }
    } else {
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < args.volume; ++idx) {
        const Legion::Point<DIM> point = args.pitches.unflatten(idx, args.rect.lo);
        args.out[point]                = args.value;
      }
    }
  }
#endif
#if defined(LEGATE_USE_CUDA) && defined(__CUDACC__)
  template <int DIM>
  static void dispatch_gpu(const Legion::Task* task,
                           const std::vector<Legion::PhysicalRegion>& regions,
                           LegateDeserializer& derez)
  {
    DeserializedArgs<DIM> args;
    args.deserialize(derez, task, regions);
    if (args.volume == 0) return;
    const size_t blocks = (args.volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_fill<DIM, DeserializedArgs<DIM>><<<blocks, THREADS_PER_BLOCK>>>(args);
  }
#elif defined(LEGATE_USE_CUDA)
  template <int DIM>
  static void dispatch_gpu(const Legion::Task* task,
                           const std::vector<Legion::PhysicalRegion>& regions,
                           LegateDeserializer& derez);
#endif
};

}  // namespace numpy
}  // namespace legate

#endif  // __NUMPY_FILL_H__
