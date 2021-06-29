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

#ifndef __NUMPY_ZIP_H__
#define __NUMPY_ZIP_H__

#include <utility>
#include "point_task.h"

namespace legate {
namespace numpy {

template <int N>
struct point_ctor {
};
template <>
struct point_ctor<1> {
  inline Legion::Point<1> operator()(coord_t x) const { return Legion::Point<1>(x); }
};
template <>
struct point_ctor<2> {
  inline Legion::Point<2> operator()(coord_t x, coord_t y) const { return Legion::Point<2>(x, y); }
};
template <>
struct point_ctor<3> {
  inline Legion::Point<3> operator()(coord_t x, coord_t y, coord_t z) const
  {
    return Legion::Point<3>(x, y, z);
  }
};
template <>
struct point_ctor<4> {
  inline Legion::Point<4> operator()(coord_t x, coord_t y, coord_t z, coord_t w) const
  {
    return Legion::Point<4>(x, y, z, w);
  }
};

#if defined(LEGATE_USE_CUDA) && defined(__CUDACC__)
template <int DIM, int N, typename Args, size_t... Is>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  gpu_zip(const Args args, const bool dense, std::index_sequence<Is...>)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= args.volume) return;
  if (dense) {
    args.outptr[idx] = Legion::Point<N>(args.inptr[Is][idx]...);
  } else {
    const Legion::Point<DIM> point = args.pitches.unflatten(idx, args.rect.lo);
    args.out[point]                = Legion::Point<N>(args.in[Is][point]...);
  }
}
#endif

template <int N>
class ZipTask : public PointTask<ZipTask<N>> {
 public:
  static const int TASK_ID = NumpyTaskOffset::NUMPY_ZIP_OFFSET + N;

  // out_region = op( in_region1, ..., in_regionN )
  static const int REGIONS = N + 1;

  template <int DIM>
  struct DeserializedArgs {
    Legion::Rect<DIM> rect;
    AccessorWO<Legion::Point<N>, DIM> out;
    AccessorRO<coord_t, DIM> in[N];
    Pitches<DIM - 1> pitches;
    size_t volume;
    Legion::Point<N>* outptr;
    const coord_t* inptr[N];
    bool deserialize(LegateDeserializer& derez,
                     const Legion::Task* task,
                     const std::vector<Legion::PhysicalRegion>& regions)
    {
      rect = NumPyProjectionFunctor::unpack_shape<DIM>(task, derez);
      out  = derez.unpack_accessor_WO<Legion::Point<N>, DIM>(regions[0], rect);
      for (int i = 0; i < N; ++i) {
        in[i] = derez.unpack_accessor_RO<coord_t, DIM>(regions[i + 1], rect);
      }
      volume = pitches.flatten(rect);
#ifndef LEGION_BOUNDS_CHECKS
      // Check to see if this is dense or not
      if (!out.accessor.is_dense_row_major(rect)) { return false; }
      outptr = out.ptr(rect);
      for (int i = 0; i < N; ++i) {
        if (!in[i].accessor.is_dense_row_major(rect)) { return false; }
        inptr[i] = in[i].ptr(rect);
      }
      return true;
#else
      // No dense execution if we're doing bounds checks
      return false;
#endif
    }
  };

  template <int DIM, size_t... Is>
  static void dispatch_cpu_impl(const Legion::Task* task,
                                const std::vector<Legion::PhysicalRegion>& regions,
                                LegateDeserializer& derez,
                                std::index_sequence<Is...>)
  {
    DeserializedArgs<DIM> args;
    const bool dense = args.deserialize(derez, task, regions);
    if (args.volume == 0) return;
    if (dense) {
      for (size_t idx = 0; idx < args.volume; ++idx) {
        args.outptr[idx] = Legion::Point<N>(args.inptr[Is][idx]...);
      }
    } else {
      CPULoop<DIM>::generic_loop(args.rect, point_ctor<N>(), args.out, args.in[Is]...);
    }
  }

  template <int DIM>
  static void dispatch_cpu(const Legion::Task* task,
                           const std::vector<Legion::PhysicalRegion>& regions,
                           LegateDeserializer& derez)
  {
    dispatch_cpu_impl<DIM>(task, regions, derez, std::make_index_sequence<N>());
  }

#ifdef LEGATE_USE_OPENMP
  template <int DIM, size_t... Is>
  static void dispatch_omp_impl(const Legion::Task* task,
                                const std::vector<Legion::PhysicalRegion>& regions,
                                LegateDeserializer& derez,
                                std::index_sequence<Is...>)
  {
    DeserializedArgs<DIM> args;
    const bool dense = args.deserialize(derez, task, regions);
    if (args.volume == 0) return;
    if (dense) {
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < args.volume; ++idx) {
        args.outptr[idx] = Legion::Point<N>(args.inptr[Is][idx]...);
      }
    } else {
      OMPLoop<DIM>::generic_loop(args.rect, point_ctor<N>(), args.out, args.in[Is]...);
    }
  }

  template <int DIM>
  static void dispatch_omp(const Legion::Task* task,
                           const std::vector<Legion::PhysicalRegion>& regions,
                           LegateDeserializer& derez)
  {
    dispatch_omp_impl<DIM>(task, regions, derez, std::make_index_sequence<N>());
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
    gpu_zip<DIM, N><<<blocks, THREADS_PER_BLOCK>>>(args, dense, std::make_index_sequence<N>());
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

#endif  // __NUMPY_ZIP_H__
