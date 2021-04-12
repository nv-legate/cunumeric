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

#include "contains.h"
#include "cuda_help.h"
#include "proj.h"

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_contains_1d(DeferredValue<bool> result, const AccessorRO<T, 1> in, const Point<1> origin, const size_t max,
                       const T value) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset;
  if (in[x] == value) result = true;
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_contains_2d(DeferredValue<bool> result, const AccessorRO<T, 2> in, const Point<2> origin, const Point<1> pitch,
                       const size_t max, const T value) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + offset % pitch[0];
  if (in[x][y] == value) result = true;
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_contains_3d(DeferredValue<bool> result, const AccessorRO<T, 3> in, const Point<3> origin, const Point<2> pitch,
                       const size_t max, const T value) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z = origin[2] + (offset % pitch[0]) % pitch[1];
  if (in[x][y][z] == value) result = true;
}

template<typename T>
/*static*/ DeferredValue<bool> ContainsTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions,
                                                            Context ctx, Runtime* runtime) {
  assert(task->futures.size() == 1);
  const T             value = task->futures[0].template get_result<T>();
  LegateDeserializer  derez(task->args, task->arglen);
  const int           dim = derez.unpack_dimension();
  DeferredValue<bool> result(false /*initial value*/);
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in     = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
      const size_t           volume = rect.volume();
      const size_t           blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      legate_contains_1d<T><<<blocks, THREADS_PER_BLOCK>>>(result, in, rect.lo, volume, value);
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2> in     = derez.unpack_accessor_RO<T, 2>(regions[0], rect);
      const size_t           volume = rect.volume();
      const size_t           blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t          pitch  = rect.hi[1] - rect.lo[1] + 1;
      legate_contains_2d<T><<<blocks, THREADS_PER_BLOCK>>>(result, in, rect.lo, Point<1>(pitch), volume, value);
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3> in       = derez.unpack_accessor_RO<T, 3>(regions[0], rect);
      const size_t           volume   = rect.volume();
      const size_t           blocks   = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t          diffy    = rect.hi[1] - rect.lo[1] + 1;
      const coord_t          diffz    = rect.hi[2] - rect.lo[2] + 1;
      const coord_t          pitch[2] = {diffy * diffz, diffz};
      legate_contains_3d<T><<<blocks, THREADS_PER_BLOCK>>>(result, in, rect.lo, Point<2>(pitch), volume, value);
      break;
    }
    default:
      assert(false);
  }
  return result;
}

INSTANTIATE_DEFERRED_VALUE_TASK_VARIANT(ContainsTask, bool, gpu_variant)

}    // namespace numpy
}    // namespace legate
