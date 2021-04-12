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

#include "arg.h"
#include "cuda_help.h"
#include "proj.h"

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_get_arg_1d(const AccessorWO<int64_t, 1> out, const AccessorRO<Argval<T>, 1> in, const Point<1> origin,
                      const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset;
  out[x]          = in[x].arg;
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_get_arg_2d(const AccessorWO<int64_t, 2> out, const AccessorRO<Argval<T>, 2> in, const Point<2> origin,
                      const Point<1> pitch, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + offset % pitch[0];
  out[x][y]       = in[x][y].arg;
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_get_arg_3d(const AccessorWO<int64_t, 3> out, const AccessorRO<Argval<T>, 3> in, const Point<3> origin,
                      const Point<2> pitch, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z = origin[2] + (offset % pitch[0]) % pitch[1];
  out[x][y][z]    = in[x][y][z].arg;
}

template<typename T>
/*static*/ void GetargTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                           Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          extra_dim = derez.unpack_dimension();
  const int          dim       = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<int64_t, 1>   out = derez.unpack_accessor_WO<int64_t, 1>(regions[0], rect);
      const AccessorRO<Argval<T>, 1> in  = (extra_dim >= 0) ? derez.unpack_accessor_RO<Argval<T>, 1>(regions[1], rect, extra_dim, 0)
                                                           : derez.unpack_accessor_RO<Argval<T>, 1>(regions[1], rect);
      const size_t volume = rect.volume();
      const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      legate_get_arg_1d<T><<<blocks, THREADS_PER_BLOCK>>>(out, in, rect.lo, volume);
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<int64_t, 2>   out = derez.unpack_accessor_WO<int64_t, 2>(regions[0], rect);
      const AccessorRO<Argval<T>, 2> in  = (extra_dim >= 0) ? derez.unpack_accessor_RO<Argval<T>, 2>(regions[1], rect, extra_dim, 0)
                                                           : derez.unpack_accessor_RO<Argval<T>, 2>(regions[1], rect);
      const size_t  volume = rect.volume();
      const size_t  blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t pitch  = rect.hi[1] - rect.lo[1] + 1;
      legate_get_arg_2d<T><<<blocks, THREADS_PER_BLOCK>>>(out, in, rect.lo, Point<1>(pitch), volume);
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<int64_t, 3>   out = derez.unpack_accessor_WO<int64_t, 3>(regions[0], rect);
      const AccessorRO<Argval<T>, 3> in  = (extra_dim >= 0) ? derez.unpack_accessor_RO<Argval<T>, 3>(regions[1], rect, extra_dim, 0)
                                                           : derez.unpack_accessor_RO<Argval<T>, 3>(regions[1], rect);
      const size_t  volume   = rect.volume();
      const size_t  blocks   = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t diffy    = rect.hi[1] - rect.lo[1] + 1;
      const coord_t diffz    = rect.hi[2] - rect.lo[2] + 1;
      const coord_t pitch[2] = {diffy * diffz, diffz};
      legate_get_arg_3d<T><<<blocks, THREADS_PER_BLOCK>>>(out, in, rect.lo, Point<2>(pitch), volume);
      break;
    }
    default:
      assert(false);
  }
}

INSTANTIATE_TASK_VARIANT(GetargTask, gpu_variant)

}    // namespace numpy
}    // namespace legate
