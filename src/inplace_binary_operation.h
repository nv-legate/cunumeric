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

#ifndef __NUMPY_INPLACE_BINARY_OPERATION_H__
#define __NUMPY_INPLACE_BINARY_OPERATION_H__

#include "point_task.h"

namespace legate {
namespace numpy {

#if defined(LEGATE_USE_CUDA) && defined(__CUDACC__)
template<int DIM, typename BinaryFunction, typename Args>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) gpu_inplace_binary_op(const Args args, const bool dense) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= args.volume) return;
  BinaryFunction func;
  if (dense) {
    args.inoutptr[idx] = func(args.inoutptr[idx], args.inptr[idx]);
  } else {
    const Legion::Point<DIM> point = args.pitches.unflatten(idx, args.rect.lo);
    args.inout[point]              = func(args.inout[point], args.in[point]);
  }
}
#endif

// Base class for all Legate's inplace binary operation tasks
template<typename Derived, typename BinaryFunction>
class InplaceBinaryOperationTask : public PointTask<Derived> {
private:
  using first_argument_type  = typename BinaryFunction::first_argument_type;
  using second_argument_type = typename BinaryFunction::second_argument_type;
  using result_type          = std::result_of_t<BinaryFunction(first_argument_type, second_argument_type)>;

public:
  static const int TASK_ID =
      task_id<BinaryFunction::op_code, NUMPY_INPLACE_VARIANT_OFFSET, result_type, first_argument_type, second_argument_type>;

  // inout_region = op(inout_region, in_region)
  static const int REGIONS = 2;

  template<int N>
  struct DeserializedArgs {
    Legion::Rect<N>                     rect;
    AccessorRW<first_argument_type, N>  inout;
    AccessorRO<second_argument_type, N> in;
    Pitches<N - 1>                      pitches;
    size_t                              volume;
    first_argument_type*                inoutptr;
    const second_argument_type*         inptr;

    bool deserialize(LegateDeserializer& derez, const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions) {
      rect   = NumPyProjectionFunctor::unpack_shape<N>(task, derez);
      inout  = derez.unpack_accessor_RW<first_argument_type, N>(regions[0], rect);
      in     = derez.unpack_accessor_RO<first_argument_type, N>(regions[1], rect);
      volume = pitches.flatten(rect);
#ifndef LEGION_BOUNDS_CHECKS
      // Check to see if this is dense or not
      return inout.accessor.is_dense_row_major(rect) && in.accessor.is_dense_row_major(rect) && (inoutptr = inout.ptr(rect)) &&
             (inptr = in.ptr(rect));
#else
      // No dense execution if we're doing bounds checks
      return false;
#endif
    }
  };

  template<int DIM>
  static void dispatch_cpu(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions,
                           LegateDeserializer& derez) {
    DeserializedArgs<DIM> args;
    const bool            dense = args.deserialize(derez, task, regions);
    if (args.volume == 0) return;
    BinaryFunction func;
    if (dense) {
      for (size_t idx = 0; idx < args.volume; ++idx)
        args.inoutptr[idx] = func(args.inoutptr[idx], args.inptr[idx]);
    } else {
      CPULoop<DIM>::binary_inplace(func, args.inout, args.in, args.rect);
    }
  }

#ifdef LEGATE_USE_OPENMP
  template<int DIM>
  static void dispatch_omp(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions,
                           LegateDeserializer& derez) {
    DeserializedArgs<DIM> args;
    const bool            dense = args.deserialize(derez, task, regions);
    if (args.volume == 0) return;
    BinaryFunction func;
    if (dense) {
#  pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < args.volume; ++idx)
        args.inoutptr[idx] = func(args.inoutptr[idx], args.inptr[idx]);
    } else {
      OMPLoop<DIM>::binary_inplace(func, args.inout, args.in, args.rect);
    }
  }
#endif
#if defined(LEGATE_USE_CUDA) && defined(__CUDACC__)
  template<int DIM>
  static void dispatch_gpu(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions,
                           LegateDeserializer& derez) {
    DeserializedArgs<DIM> args;
    const bool            dense = args.deserialize(derez, task, regions);
    if (args.volume == 0) return;
    const size_t blocks = (args.volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_inplace_binary_op<DIM, BinaryFunction, DeserializedArgs<DIM>><<<blocks, THREADS_PER_BLOCK>>>(args, dense);
  }
#elif defined(LEGATE_USE_CUDA)
  template<int DIM>
  static void dispatch_gpu(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, LegateDeserializer& derez);
#endif
};

}    // namespace numpy
}    // namespace legate

#endif    // __NUMPY_INPLACE_BINARY_OPERATION_H__
