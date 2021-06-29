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

#ifndef __NUMPY_TRANSFORM_H__
#define __NUMPY_TRANSFORM_H__

#include "point_task.h"

namespace legate {
namespace numpy {

template <int M, int N>
struct TransformOperation {
  __CUDA_HD__ constexpr Legion::Point<M> operator()(
    const Legion::Point<N>& p, const Legion::AffineTransform<M, N>& transform) const
  {
    return transform[p];
  }
};

#if defined(LEGATE_USE_CUDA) && defined(__CUDACC__)
template <int DIM, int M, int N, typename Args>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  gpu_transform(const Args args, const bool dense)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= args.volume) return;
  TransformOperation<M, N> func;
  if (dense) {
    args.outptr[idx] = func(args.inptr[idx], args.transform);
  } else {
    const Legion::Point<DIM> point = args.pitches.unflatten(idx, args.rect.lo);
    args.out[point]                = func(args.in[point], args.transform);
  }
}
#endif

template <int M, int N>
class TransformTask : public PointTask<TransformTask<M, N>> {
 public:
  static const int TASK_ID = NUMPY_TRANSFORM_OFFSET + M * (LEGION_MAX_DIM + 1) + N;

  // out_region = op in_region;
  static const int REGIONS = 2;

  template <int DIM>
  struct DeserializedArgs {
    Legion::Rect<DIM> rect;
    AccessorWO<Legion::Point<M>, DIM> out;
    AccessorRO<Legion::Point<N>, DIM> in;
    Pitches<DIM - 1> pitches;
    size_t volume;
    Legion::AffineTransform<M, N> transform;
    Legion::Point<M>* outptr;
    const Legion::Point<N>* inptr;
    bool deserialize(LegateDeserializer& derez,
                     const Legion::Task* task,
                     const std::vector<Legion::PhysicalRegion>& regions)
    {
      rect   = NumPyProjectionFunctor::unpack_shape<DIM>(task, derez);
      out    = derez.unpack_accessor_WO<Legion::Point<M>, DIM>(regions[0], rect);
      in     = derez.unpack_accessor_RO<Legion::Point<N>, DIM>(regions[1], rect);
      volume = pitches.flatten(rect);
      size_t size;
      const void* data = task->futures[0].get_buffer(
        Realm::Memory::SYSTEM_MEM, &size, false /*check_extent*/, true /*silence_warnings*/);
      transform = LegateDeserializer(data, size).unpack_transform<M, N>();
#ifndef LEGION_BOUNDS_CHECKS
      // Check to see if this is dense or not
      return out.accessor.is_dense_row_major(rect) && in.accessor.is_dense_row_major(rect) &&
             (outptr = out.ptr(rect)) && (inptr = in.ptr(rect));
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
    TransformOperation<M, N> func;
    if (dense) {
      for (size_t idx = 0; idx < args.volume; ++idx) {
        args.outptr[idx] = func(args.inptr[idx], args.transform);
      }
    } else {
      CPULoop<DIM>::unary_loop(func, args.out, args.in, args.rect, args.transform);
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
    TransformOperation<M, N> func;
    if (dense) {
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < args.volume; ++idx) {
        args.outptr[idx] = func(args.inptr[idx], args.transform);
      }
    } else {
      OMPLoop<DIM>::unary_loop(func, args.out, args.in, args.rect, args.transform);
    }
  }
#endif  // LEGATE_USE_OPENMP

#if defined(LEGATE_USE_CUDA) && defined(__CUDACC__)
  template <int DIM>
  static void dispatch_gpu(const Legion::Task* task,
                           const std::vector<Legion::PhysicalRegion>& regions,
                           LegateDeserializer& derez)
  {
    DeserializedArgs<DIM> args;
    const bool dense = args.deserialize(derez, task, regions);
    if (args.volume == 0) return;
    const size_t blocks = (args.volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_transform<DIM, M, N><<<blocks, THREADS_PER_BLOCK>>>(args, dense);
  }
#elif defined(LEGATE_USE_CUDA)
  template <int DIM>
  static void dispatch_gpu(const Legion::Task* task,
                           const std::vector<Legion::PhysicalRegion>& regions,
                           LegateDeserializer& derez);
#endif  // LEGATE_USE_CUDA
};

}  // namespace numpy
}  // namespace legate

#endif  // __NUMPY_TRANSFORM_H__
