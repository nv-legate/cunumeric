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
#include "proj.h"
#include "tile.h"

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_tile_1d(const AccessorWO<T, 1> out, const AccessorRO<T, 1> in, const Point<1> bounds, const Point<1> origin,
                   const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset;
  out[x]          = in[x % bounds[0]];
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_tile_2d_1d(const AccessorWO<T, 2> out, const AccessorRO<T, 1> in, const Point<1> bounds, const Point<2> origin,
                      const Point<1> pitch, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + offset % pitch[0];
  out[x][y]       = in[y % bounds[0]];
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_tile_2d_2d(const AccessorWO<T, 2> out, const AccessorRO<T, 2> in, const Point<2> bounds, const Point<2> origin,
                      const Point<1> pitch, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + offset % pitch[0];
  out[x][y]       = in[x % bounds[0]][y % bounds[1]];
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_tile_3d_1d(const AccessorWO<T, 3> out, const AccessorRO<T, 1> in, const Point<1> bounds, const Point<3> origin,
                      const Point<2> pitch, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z = origin[2] + (offset % pitch[0]) % pitch[1];
  out[x][y][z]    = in[z % bounds[0]];
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_tile_3d_2d(const AccessorWO<T, 3> out, const AccessorRO<T, 2> in, const Point<2> bounds, const Point<3> origin,
                      const Point<2> pitch, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z = origin[2] + (offset % pitch[0]) % pitch[1];
  out[x][y][z]    = in[y % bounds[0]][z % bounds[1]];
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_tile_3d_3d(const AccessorWO<T, 3> out, const AccessorRO<T, 3> in, const Point<3> bounds, const Point<3> origin,
                      const Point<2> pitch, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z = origin[2] + (offset % pitch[0]) % pitch[1];
  out[x][y][z]    = in[x % bounds[0]][y % bounds[1]][z % bounds[2]];
}

template<typename T>
/*static*/ void TileTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                         Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dst_dim = derez.unpack_dimension();
  switch (dst_dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 1> out    = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
      const Point<1>         bounds = derez.unpack_point<1>();
      const AccessorRO<T, 1> in =
          derez.unpack_accessor_RO<T, 1>(regions[1], Rect<1>(Point<1>::ZEROES(), bounds - Point<1>::ONES()));
      const size_t volume = rect.volume();
      const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      legate_tile_1d<T><<<blocks, THREADS_PER_BLOCK>>>(out, in, bounds, rect.lo, volume);
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 2> out     = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
      const size_t           volume  = rect.volume();
      const size_t           blocks  = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t          pitch   = rect.hi[1] - rect.lo[1] + 1;
      const int              src_dim = derez.unpack_dimension();
      switch (src_dim) {
        case 1: {
          const Point<1>         bounds = derez.unpack_point<1>();
          const AccessorRO<T, 1> in =
              derez.unpack_accessor_RO<T, 1>(regions[1], Rect<1>(Point<1>::ZEROES(), bounds - Point<1>::ONES()));
          legate_tile_2d_1d<T><<<blocks, THREADS_PER_BLOCK>>>(out, in, bounds, rect.lo, pitch, volume);
          break;
        }
        case 2: {
          const Point<2>         bounds = derez.unpack_point<2>();
          const AccessorRO<T, 2> in =
              derez.unpack_accessor_RO<T, 2>(regions[1], Rect<2>(Point<2>::ZEROES(), bounds - Point<2>::ONES()));
          legate_tile_2d_2d<T><<<blocks, THREADS_PER_BLOCK>>>(out, in, bounds, rect.lo, pitch, volume);
          break;
        }
        default:
          assert(false);
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 3> out      = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
      const size_t           volume   = rect.volume();
      const size_t           blocks   = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t          diffy    = rect.hi[1] - rect.lo[1] + 1;
      const coord_t          diffz    = rect.hi[2] - rect.lo[2] + 1;
      const coord_t          pitch[2] = {diffy * diffz, diffz};
      const int              src_dim  = derez.unpack_dimension();
      switch (src_dim) {
        case 1: {
          const Point<1>         bounds = derez.unpack_point<1>();
          const AccessorRO<T, 1> in =
              derez.unpack_accessor_RO<T, 1>(regions[1], Rect<1>(Point<1>::ZEROES(), bounds - Point<1>::ONES()));
          legate_tile_3d_1d<T><<<blocks, THREADS_PER_BLOCK>>>(out, in, bounds, rect.lo, Point<2>(pitch), volume);
          break;
        }
        case 2: {
          const Point<2>         bounds = derez.unpack_point<2>();
          const AccessorRO<T, 2> in =
              derez.unpack_accessor_RO<T, 2>(regions[1], Rect<2>(Point<2>::ZEROES(), bounds - Point<2>::ONES()));
          legate_tile_3d_2d<T><<<blocks, THREADS_PER_BLOCK>>>(out, in, bounds, rect.lo, Point<2>(pitch), volume);
          break;
        }
        case 3: {
          const Point<3>         bounds = derez.unpack_point<3>();
          const AccessorRO<T, 3> in =
              derez.unpack_accessor_RO<T, 3>(regions[1], Rect<3>(Point<3>::ZEROES(), bounds - Point<3>::ONES()));
          legate_tile_3d_3d<T><<<blocks, THREADS_PER_BLOCK>>>(out, in, bounds, rect.lo, Point<2>(pitch), volume);
          break;
        }
        default:
          assert(false);
      }
      break;
    }
    default:
      assert(false);
  }
}

INSTANTIATE_TASK_VARIANT(TileTask, gpu_variant)

}    // namespace numpy
}    // namespace legate
