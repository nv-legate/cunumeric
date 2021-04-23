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

#ifndef __NUMPY_BROADCAST_BINARY_OPERATION_H__
#define __NUMPY_BROADCAST_BINARY_OPERATION_H__

#include "point_task.h"

namespace legate {
namespace numpy {

#if defined(LEGATE_USE_CUDA) && defined(__CUDACC__)
template <int DIM, typename BinaryFunction, typename Args>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  gpu_broadcast_binary_op(const Args args, const bool dense)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= args.volume) return;
  BinaryFunction func;
  if (dense) {
    args.outptr[idx] = func(args.inptr[idx], args.scalar);
  } else {
    const Legion::Point<DIM> point = args.pitches.unflatten(idx, args.rect.lo);
    args.out[point]                = func(args.in[point], args.scalar);
  }
}
#endif

// Base class for all Legate's binary operation tasks
template <class Derived, class BinaryFunction>
class BroadcastBinaryOperationTask : public PointTask<Derived> {
 private:
  using first_argument_type  = typename BinaryFunction::first_argument_type;
  using second_argument_type = typename BinaryFunction::second_argument_type;
  using result_type = std::result_of_t<BinaryFunction(first_argument_type, second_argument_type)>;

 public:
  static_assert(std::is_same<first_argument_type, second_argument_type>::value,
                "BroadcastBinaryOperation currently requires first_argument_type and "
                "second_argument_type to be the same type.");
  static const int TASK_ID = task_id<BinaryFunction::op_code,
                                     NUMPY_BROADCAST_VARIANT_OFFSET,
                                     result_type,
                                     first_argument_type,
                                     second_argument_type>;

  // out_region = in_region1 op scalar
  static const int REGIONS = 2;

  template <int N>
  struct DeserializedArgs {
    Legion::Rect<N> rect;
    AccessorWO<result_type, N> out;
    AccessorRO<first_argument_type, N> in;
    Pitches<N - 1> pitches;
    size_t volume;
    second_argument_type scalar;
    result_type* outptr;
    const first_argument_type* inptr;

    bool deserialize(LegateDeserializer& derez,
                     const Legion::Task* task,
                     const std::vector<Legion::PhysicalRegion>& regions)
    {
      rect   = NumPyProjectionFunctor::unpack_shape<N>(task, derez);
      out    = derez.unpack_accessor_WO<result_type, N>(regions[0], rect);
      in     = derez.unpack_accessor_RO<first_argument_type, N>(regions[1], rect);
      scalar = task->futures[0].get_result<second_argument_type>(true /*silence warnings*/);
      volume = pitches.flatten(rect);
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
    BinaryFunction func;
    if (dense) {
      for (size_t idx = 0; idx < args.volume; ++idx)
        args.outptr[idx] = func(args.inptr[idx], args.scalar);
    } else {
      const Scalar<second_argument_type, DIM> scalar(args.scalar);
      CPULoop<DIM>::binary_loop(func, args.out, args.in, scalar, args.rect);
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
    BinaryFunction func;
    if (dense) {
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < args.volume; ++idx)
        args.outptr[idx] = func(args.inptr[idx], args.scalar);
    } else {
      const Scalar<second_argument_type, DIM> scalar(args.scalar);
      OMPLoop<DIM>::binary_loop(func, args.out, args.in, scalar, args.rect);
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
    const bool dense = args.deserialize(derez, task, regions);
    if (args.volume == 0) return;
    const size_t blocks = (args.volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_broadcast_binary_op<DIM, BinaryFunction, DeserializedArgs<DIM>>
      <<<blocks, THREADS_PER_BLOCK>>>(args, dense);
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

#endif  // __NUMPY_BROADCAST_BINARY_OPERATION_H__
