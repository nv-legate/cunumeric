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

#include "close.h"
#include "cuda_help.h"
#include "proj.h"

template<typename T>
__host__ __device__ T abs(const T& x) {
  return static_cast<double>(x) < 0.0 ? -x : x;
}
// #define FABS(x) ((x < T{0}) ? -(x) : (x))
#define FABS(x) (abs(x))

using namespace Legion;

namespace legate {
namespace numpy {

__device__ __forceinline__ static double operator*(double lhs, __half rhs) { return (lhs * ((double)rhs)); }

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_close_1d(DeferredValue<bool> result, const AccessorRO<T, 1> in1, const AccessorRO<T, 1> in2, const double rtol,
                    const double atol, const Point<1> origin, const size_t max) {
  const size_t  offset = blockIdx.x * blockDim.x + threadIdx.x;
  const coord_t x      = origin[0] + offset;
  int           value =
      (offset >= max) ? 1 : (static_cast<double>(FABS(in1[x] - in2[x])) <= static_cast<double>(atol + rtol * FABS(in2[x]))) ? 1 : 0;
  reduce_bool(result, value);
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_close_2d(DeferredValue<bool> result, const AccessorRO<T, 2> in1, const AccessorRO<T, 2> in2, const double rtol,
                    const double atol, const Point<2> origin, const Point<1> pitch, const size_t max) {
  const size_t  offset = blockIdx.x * blockDim.x + threadIdx.x;
  const coord_t x      = origin[0] + offset / pitch[0];
  const coord_t y      = origin[1] + offset % pitch[0];
  int           value =
      (offset >= max)
          ? 1
          : (static_cast<double>(FABS(in1[x][y] - in2[x][y])) <= static_cast<double>(atol + rtol * FABS(in2[x][y]))) ? 1 : 0;
  reduce_bool(result, value);
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_close_3d(DeferredValue<bool> result, const AccessorRO<T, 3> in1, const AccessorRO<T, 3> in2, const double rtol,
                    const double atol, const Point<3> origin, const Point<2> pitch, const size_t max) {
  const size_t  offset = blockIdx.x * blockDim.x + threadIdx.x;
  const coord_t x      = origin[0] + offset / pitch[0];
  const coord_t y      = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z      = origin[2] + (offset % pitch[0]) % pitch[1];
  int           value =
      (offset >= max)
          ? 1
          : (static_cast<double>(FABS(in1[x][y][z] - in2[x][y][z])) <= static_cast<double>(atol + rtol * FABS(in2[x][y][z]))) ? 1
                                                                                                                              : 0;
  reduce_bool(result, value);
}

template<typename T>
/*static*/ DeferredValue<bool> CloseTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                         Runtime* runtime) {
  assert(task->futures.size() == 2);
  const double        rtol = task->futures[0].get_result<double>();
  const double        atol = task->futures[1].get_result<double>();
  DeferredValue<bool> result(true /*initial value*/);
  LegateDeserializer  derez(task->args, task->arglen);
  const int           dim = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in1    = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
      const AccessorRO<T, 1> in2    = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
      const size_t           volume = rect.volume();
      const size_t           blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      legate_close_1d<T><<<blocks, THREADS_PER_BLOCK>>>(result, in1, in2, rtol, atol, rect.lo, volume);
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2> in1    = derez.unpack_accessor_RO<T, 2>(regions[0], rect);
      const AccessorRO<T, 2> in2    = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      const size_t           volume = rect.volume();
      const size_t           blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t          pitch  = rect.hi[1] - rect.lo[1] + 1;
      legate_close_2d<T><<<blocks, THREADS_PER_BLOCK>>>(result, in1, in2, rtol, atol, rect.lo, Point<1>(pitch), volume);
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3> in1      = derez.unpack_accessor_RO<T, 3>(regions[0], rect);
      const AccessorRO<T, 3> in2      = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      const size_t           volume   = rect.volume();
      const size_t           blocks   = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t          diffy    = rect.hi[1] - rect.lo[1] + 1;
      const coord_t          diffz    = rect.hi[2] - rect.lo[2] + 1;
      const coord_t          pitch[2] = {diffy * diffz, diffz};
      legate_close_3d<T><<<blocks, THREADS_PER_BLOCK>>>(result, in1, in2, rtol, atol, rect.lo, Point<2>(pitch), volume);
      break;
    }
    default:
      assert(false);
  }
  return result;
}

INSTANTIATE_DEFERRED_VALUE_TASK_VARIANT(CloseTask, bool, gpu_variant)

template<typename T, bool FIRST>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_close_broadcast_1d(DeferredValue<bool> result, const AccessorRO<T, 1> in1, const T in2, const double rtol,
                              const double atol, const Point<1> origin, const size_t max) {
  const size_t  offset = blockIdx.x * blockDim.x + threadIdx.x;
  const coord_t x      = origin[0] + offset;
  int           value  = (offset >= max)
                  ? 1
                  : FIRST ? ((static_cast<double>(FABS(in2 - in1[x])) <= static_cast<double>(atol + rtol * FABS(in1[x]))) ? 1 : 0)
                          : ((static_cast<double>(FABS(in1[x] - in2)) <= static_cast<double>(atol + rtol * FABS(in2))) ? 1 : 0);
  reduce_bool(result, value);
}

template<typename T, bool FIRST>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_close_broadcast_2d(DeferredValue<bool> result, const AccessorRO<T, 2> in1, const T in2, const double rtol,
                              const double atol, const Point<2> origin, const Point<1> pitch, const size_t max) {
  const size_t  offset = blockIdx.x * blockDim.x + threadIdx.x;
  const coord_t x      = origin[0] + offset / pitch[0];
  const coord_t y      = origin[1] + offset % pitch[0];
  int           value =
      (offset >= max)
          ? 1
          : FIRST ? ((static_cast<double>(FABS(in2 - in1[x][y])) <= static_cast<double>(atol + rtol * FABS(in1[x][y]))) ? 1 : 0)
                  : ((static_cast<double>(FABS(in1[x][y] - in2)) <= static_cast<double>(atol + rtol * FABS(in2))) ? 1 : 0);
  reduce_bool(result, value);
}

template<typename T, bool FIRST>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_close_broadcast_3d(DeferredValue<bool> result, const AccessorRO<T, 3> in1, const T in2, const double rtol,
                              const double atol, const Point<3> origin, const Point<2> pitch, const size_t max) {
  const size_t  offset = blockIdx.x * blockDim.x + threadIdx.x;
  const coord_t x      = origin[0] + offset / pitch[0];
  const coord_t y      = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z      = origin[2] + (offset % pitch[0]) % pitch[1];
  int           value =
      (offset >= max)
          ? 1
          : FIRST
                ? ((static_cast<double>(FABS(in2 - in1[x][y][z])) <= static_cast<double>(atol + rtol * FABS(in1[x][y][z]))) ? 1 : 0)
                : ((static_cast<double>(FABS(in1[x][y][z] - in2)) <= static_cast<double>(atol + rtol * FABS(in2))) ? 1 : 0);
  reduce_bool(result, value);
}

template<typename T>
/*static*/ DeferredValue<bool> CloseBroadcast<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions,
                                                              Context ctx, Runtime* runtime) {
  DeferredValue<bool> result(true /*initial value*/);
  LegateDeserializer  derez(task->args, task->arglen);
  const int           dim = derez.unpack_dimension();
  assert(task->futures.size() == 3);
  const T      in2  = task->futures[0].get_result<T>();
  const double rtol = task->futures[1].get_result<double>();
  const double atol = task->futures[2].get_result<double>();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in1   = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
      const unsigned         index = derez.unpack_32bit_uint();
      assert((index == 0) || (index == 1));
      const size_t volume = rect.volume();
      const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      if (index == 0)
        legate_close_broadcast_1d<T, true><<<blocks, THREADS_PER_BLOCK>>>(result, in1, in2, rtol, atol, rect.lo, volume);
      else
        legate_close_broadcast_1d<T, false><<<blocks, THREADS_PER_BLOCK>>>(result, in1, in2, rtol, atol, rect.lo, volume);
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2> in1   = derez.unpack_accessor_RO<T, 2>(regions[0], rect);
      const unsigned         index = derez.unpack_32bit_uint();
      assert((index == 0) || (index == 1));
      const size_t  volume = rect.volume();
      const size_t  blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t pitch  = rect.hi[1] - rect.lo[1] + 1;
      if (index == 0)
        legate_close_broadcast_2d<T, true>
            <<<blocks, THREADS_PER_BLOCK>>>(result, in1, in2, rtol, atol, rect.lo, Point<1>(pitch), volume);
      else
        legate_close_broadcast_2d<T, false>
            <<<blocks, THREADS_PER_BLOCK>>>(result, in1, in2, rtol, atol, rect.lo, Point<1>(pitch), volume);
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3> in1   = derez.unpack_accessor_RO<T, 3>(regions[0], rect);
      const unsigned         index = derez.unpack_32bit_uint();
      assert((index == 0) || (index == 1));
      const size_t  volume   = rect.volume();
      const size_t  blocks   = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t diffy    = rect.hi[1] - rect.lo[1] + 1;
      const coord_t diffz    = rect.hi[2] - rect.lo[2] + 1;
      const coord_t pitch[2] = {diffy * diffz, diffz};
      if (index == 0)
        legate_close_broadcast_3d<T, true>
            <<<blocks, THREADS_PER_BLOCK>>>(result, in1, in2, rtol, atol, rect.lo, Point<2>(pitch), volume);
      else
        legate_close_broadcast_3d<T, false>
            <<<blocks, THREADS_PER_BLOCK>>>(result, in1, in2, rtol, atol, rect.lo, Point<2>(pitch), volume);
      break;
    }
    default:
      assert(false);
  }
  return result;
}

INSTANTIATE_DEFERRED_VALUE_TASK_VARIANT(CloseBroadcast, bool, gpu_variant)

}    // namespace numpy
}    // namespace legate
