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
#include "less_equal_reduce.h"
#include "proj.h"

using namespace Legion;

namespace legate {
namespace numpy {

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_leq_reduce_1d(DeferredValue<bool> result,
                       const AccessorRO<T, 1> in1,
                       const AccessorRO<T, 1> in2,
                       const Point<1> origin,
                       const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  const coord_t x     = origin[0] + offset;
  int value           = (offset >= max) ? 1 : (in1[x] <= in2[x]) ? 1 : 0;
  reduce_bool(result, value);
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_leq_reduce_2d(DeferredValue<bool> result,
                       const AccessorRO<T, 2> in1,
                       const AccessorRO<T, 2> in2,
                       const Point<2> origin,
                       const Point<1> pitch,
                       const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  const coord_t x     = origin[0] + offset / pitch[0];
  const coord_t y     = origin[1] + offset % pitch[0];
  int value           = (offset >= max) ? 1 : (in1[x][y] <= in2[x][y]) ? 1 : 0;
  reduce_bool(result, value);
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_leq_reduce_3d(DeferredValue<bool> result,
                       const AccessorRO<T, 3> in1,
                       const AccessorRO<T, 3> in2,
                       const Point<3> origin,
                       const Point<2> pitch,
                       const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  const coord_t x     = origin[0] + offset / pitch[0];
  const coord_t y     = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z     = origin[2] + (offset % pitch[0]) % pitch[1];
  int value           = (offset >= max) ? 1 : (in1[x][y][z] <= in2[x][y][z]) ? 1 : 0;
  reduce_bool(result, value);
}

template <typename T>
/*static*/ DeferredValue<bool> LessEqualReducTask<T>::gpu_variant(
  const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime)
{
  DeferredValue<bool> result(true /*initial value*/);
  LegateDeserializer derez(task->args, task->arglen);
  const int dim = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in1 = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
      const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
      const size_t volume        = rect.volume();
      const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      legate_leq_reduce_1d<T><<<blocks, THREADS_PER_BLOCK>>>(result, in1, in2, rect.lo, volume);
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2> in1 = derez.unpack_accessor_RO<T, 2>(regions[0], rect);
      const AccessorRO<T, 2> in2 = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      const size_t volume        = rect.volume();
      const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t pitch        = rect.hi[1] - rect.lo[1] + 1;
      legate_leq_reduce_2d<T>
        <<<blocks, THREADS_PER_BLOCK>>>(result, in1, in2, rect.lo, Point<1>(pitch), volume);
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3> in1 = derez.unpack_accessor_RO<T, 3>(regions[0], rect);
      const AccessorRO<T, 3> in2 = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      const size_t volume        = rect.volume();
      const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t diffy        = rect.hi[1] - rect.lo[1] + 1;
      const coord_t diffz        = rect.hi[2] - rect.lo[2] + 1;
      const coord_t pitch[2]     = {diffy * diffz, diffz};
      legate_leq_reduce_3d<T>
        <<<blocks, THREADS_PER_BLOCK>>>(result, in1, in2, rect.lo, Point<2>(pitch), volume);
      break;
    }
    default: assert(false);
  }
  return result;
}

INSTANTIATE_DEFERRED_VALUE_TASK_VARIANT(LessEqualReducTask, bool, gpu_variant)

}  // namespace numpy
}  // namespace legate
