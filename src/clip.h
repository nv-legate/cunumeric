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

#ifndef __NUMPY_CLIP_H__
#define __NUMPY_CLIP_H__

#include "point_task.h"

namespace legate {
namespace numpy {
template <class T>
struct ClipOperation {
  using argument_type           = T;
  constexpr static auto op_code = NumPyOpCode::NUMPY_CLIP;

  __CUDA_HD__ constexpr T operator()(const T& a, const T min, const T max) const
  {
    return (a < min) ? min : (a > max) ? max : a;
  }
};

#if defined(LEGATE_USE_CUDA) && defined(__CUDACC__)
template <int DIM, typename T, typename Args>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  gpu_clip(const Args args, const bool dense)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= args.volume) return;
  ClipOperation<T> func;
  if (dense) {
    args.outptr[idx] = func(args.inptr[idx], args.min, args.max);
  } else {
    const Legion::Point<DIM> point = args.pitches.unflatten(idx, args.rect.lo);
    args.out[point]                = func(args.in[point], args.min, args.max);
  }
}

template <int DIM, typename T, typename Args>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  gpu_clip_inplace(const Args args, const bool dense)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= args.volume) return;
  ClipOperation<T> func;
  if (dense) {
    args.inoutptr[idx] = func(args.inoutptr[idx], args.min, args.max);
  } else {
    const Legion::Point<DIM> point = args.pitches.unflatten(idx, args.rect.lo);
    args.inout[point]              = func(args.inout[point], args.min, args.max);
  }
}
#endif

// Clip is like a unary operation but with some state for its operator
template <class T>
class ClipTask : public PointTask<ClipTask<T>> {
 private:
  using argument_type = typename ClipOperation<T>::argument_type;
  using result_type   = typename ClipOperation<T>::argument_type;

 public:
  static const int TASK_ID =
    task_id<ClipOperation<T>::op_code, NUMPY_NORMAL_VARIANT_OFFSET, argument_type, result_type>;

  // out_region = op in_region;
  static const int REGIONS = 2;

  template <int N>
  struct DeserializedArgs {
    Legion::Rect<N> rect;
    AccessorWO<result_type, N> out;
    AccessorRO<argument_type, N> in;
    Pitches<N - 1> pitches;
    size_t volume;
    argument_type min;
    argument_type max;
    result_type* outptr;
    const argument_type* inptr;
    bool deserialize(LegateDeserializer& derez,
                     const Legion::Task* task,
                     const std::vector<Legion::PhysicalRegion>& regions)
    {
      rect   = NumPyProjectionFunctor::unpack_shape<N>(task, derez);
      out    = derez.unpack_accessor_WO<result_type, N>(regions[0], rect);
      in     = derez.unpack_accessor_RO<argument_type, N>(regions[1], rect);
      min    = task->futures[0].get_result<argument_type>(true /*silence warnings*/);
      max    = task->futures[1].get_result<argument_type>(true /*slience warnings*/);
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
    ClipOperation<T> func;
    if (dense) {
      for (size_t idx = 0; idx < args.volume; ++idx)
        args.outptr[idx] = func(args.inptr[idx], args.min, args.max);
    } else {
      CPULoop<DIM>::unary_loop(func, args.out, args.in, args.rect, args.min, args.max);
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
    ClipOperation<T> func;
    if (dense) {
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < args.volume; ++idx) {
        args.outptr[idx] = func(args.inptr[idx], args.min, args.max);
      }
    } else {
      OMPLoop<DIM>::unary_loop(func, args.out, args.in, args.rect, args.min, args.max);
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
    gpu_clip<DIM, T, DeserializedArgs<DIM>><<<blocks, THREADS_PER_BLOCK>>>(args, dense);
  }
#elif defined(LEGATE_USE_CUDA)
  template <int DIM>
  static void dispatch_gpu(const Legion::Task* task,
                           const std::vector<Legion::PhysicalRegion>& regions,
                           LegateDeserializer& derez);
#endif
};

template <typename T>
class ClipInplace : public PointTask<ClipInplace<T>> {
 private:
  using argument_type = typename ClipOperation<T>::argument_type;
  using result_type   = typename ClipOperation<T>::argument_type;

 public:
  static const int TASK_ID =
    task_id<ClipOperation<T>::op_code, NUMPY_INPLACE_VARIANT_OFFSET, result_type, argument_type>;

  // inout_region = op(inout_region)
  static const int REGIONS = 1;

  template <int N>
  struct DeserializedArgs {
    Legion::Rect<N> rect;
    AccessorRW<result_type, N> inout;
    Pitches<N - 1> pitches;
    size_t volume;
    argument_type min;
    argument_type max;
    argument_type* inoutptr;
    bool deserialize(LegateDeserializer& derez,
                     const Legion::Task* task,
                     const std::vector<Legion::PhysicalRegion>& regions)
    {
      rect   = NumPyProjectionFunctor::unpack_shape<N>(task, derez);
      inout  = derez.unpack_accessor_RW<result_type, N>(regions[0], rect);
      min    = task->futures[0].get_result<argument_type>(true /*silence warnings*/);
      max    = task->futures[1].get_result<argument_type>(true /*silence warnings*/);
      volume = pitches.flatten(rect);
#ifndef LEGION_BOUNDS_CHECKS
      // Check to see if this is dense or not
      return inout.accessor.is_dense_row_major(rect) && (inoutptr = inout.ptr(rect));
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
    ClipOperation<T> func;
    if (dense) {
      for (size_t idx = 0; idx < args.volume; ++idx)
        args.inoutptr[idx] = func(args.inoutptr[idx], args.min, args.max);
    } else {
      CPULoop<DIM>::unary_inplace(func, args.inout, args.rect, args.min, args.max);
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
    ClipOperation<T> func;
    if (dense) {
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < args.volume; ++idx) {
        args.inoutptr[idx] = func(args.inoutptr[idx], args.min, args.max);
      }
    } else {
      OMPLoop<DIM>::unary_inplace(func, args.inout, args.rect, args.min, args.max);
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
    gpu_clip_inplace<DIM, T, DeserializedArgs<DIM>><<<blocks, THREADS_PER_BLOCK>>>(args, dense);
  }
#elif defined(LEGATE_USE_CUDA)
  template <int DIM>
  static void dispatch_gpu(const Legion::Task* task,
                           const std::vector<Legion::PhysicalRegion>& regions,
                           LegateDeserializer& derez);
#endif
};

template <typename T>
class ClipScalar : public NumPyTask<ClipScalar<T>> {
 private:
  using argument_type = typename ClipOperation<T>::argument_type;
  using result_type   = typename ClipOperation<T>::argument_type;

 public:
  // XXX figure out how to hoist this into PointTask
  static const int TASK_ID =
    task_id<ClipOperation<T>::op_code, NUMPY_SCALAR_VARIANT_OFFSET, result_type, argument_type>;

  static const int REGIONS = 0;

  static result_type cpu_variant(const Legion::Task* task,
                                 const std::vector<Legion::PhysicalRegion>& regions,
                                 Legion::Context ctx,
                                 Legion::Runtime* runtime)
  {
    argument_type rhs = task->futures[0].get_result<argument_type>(true /*silence warnings*/);
    argument_type min = task->futures[1].get_result<argument_type>(true /*silence warnings*/);
    argument_type max = task->futures[2].get_result<argument_type>(true /*silence warnings*/);

    ClipOperation<T> func;
    return func(rhs, min, max);
  }

 private:
  struct StaticRegistrar {
    StaticRegistrar()
    {
      ClipScalar::template register_variants_with_return<result_type, argument_type>();
    }
  };

  virtual void force_instantiation_of_static_registrar() { (void)&static_registrar; }

  // this static member registers this task's variants during static initialization
  static const StaticRegistrar static_registrar;
};

// this is the definition of ScalarUnaryOperationTask::static_registrar
template <class T>
const typename ClipScalar<T>::StaticRegistrar ClipScalar<T>::static_registrar{};

}  // namespace numpy
}  // namespace legate

#endif  // __NUMPY_CLIP_H__
