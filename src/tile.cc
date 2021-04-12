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

#include "tile.h"
#include "proj.h"

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
/*static*/ void TileTask<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
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
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        out[x] = in[x % bounds[0]];
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 2> out     = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
      const int              src_dim = derez.unpack_dimension();
      switch (src_dim) {
        case 1: {
          const Point<1>         bounds = derez.unpack_point<1>();
          const AccessorRO<T, 1> in =
              derez.unpack_accessor_RO<T, 1>(regions[1], Rect<1>(Point<1>::ZEROES(), bounds - Point<1>::ONES()));
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              out[x][y] = in[y % bounds[0]];
          break;
        }
        case 2: {
          const Point<2>         bounds = derez.unpack_point<2>();
          const AccessorRO<T, 2> in =
              derez.unpack_accessor_RO<T, 2>(regions[1], Rect<2>(Point<2>::ZEROES(), bounds - Point<2>::ONES()));
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              out[x][y] = in[x % bounds[0]][y % bounds[1]];
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
      const AccessorWO<T, 3> out     = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
      const int              src_dim = derez.unpack_dimension();
      switch (src_dim) {
        case 1: {
          const Point<1>         bounds = derez.unpack_point<1>();
          const AccessorRO<T, 1> in =
              derez.unpack_accessor_RO<T, 1>(regions[1], Rect<1>(Point<1>::ZEROES(), bounds - Point<1>::ONES()));
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                out[x][y][z] = in[z % bounds[0]];
          break;
        }
        case 2: {
          const Point<2>         bounds = derez.unpack_point<2>();
          const AccessorRO<T, 2> in =
              derez.unpack_accessor_RO<T, 2>(regions[1], Rect<2>(Point<2>::ZEROES(), bounds - Point<2>::ONES()));
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                out[x][y][z] = in[y % bounds[0]][z % bounds[1]];
          break;
        }
        case 3: {
          const Point<3>         bounds = derez.unpack_point<3>();
          const AccessorRO<T, 3> in =
              derez.unpack_accessor_RO<T, 3>(regions[1], Rect<3>(Point<3>::ZEROES(), bounds - Point<3>::ONES()));
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                out[x][y][z] = in[x % bounds[0]][y % bounds[1]][z % bounds[2]];
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

#ifdef LEGATE_USE_OPENMP
template<typename T>
/*static*/ void TileTask<T>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
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
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        out[x] = in[x % bounds[0]];
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 2> out     = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
      const int              src_dim = derez.unpack_dimension();
      switch (src_dim) {
        case 1: {
          const Point<1>         bounds = derez.unpack_point<1>();
          const AccessorRO<T, 1> in =
              derez.unpack_accessor_RO<T, 1>(regions[1], Rect<1>(Point<1>::ZEROES(), bounds - Point<1>::ONES()));
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              out[x][y] = in[y % bounds[0]];
          break;
        }
        case 2: {
          const Point<2>         bounds = derez.unpack_point<2>();
          const AccessorRO<T, 2> in =
              derez.unpack_accessor_RO<T, 2>(regions[1], Rect<2>(Point<2>::ZEROES(), bounds - Point<2>::ONES()));
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              out[x][y] = in[x % bounds[0]][y % bounds[1]];
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
      const AccessorWO<T, 3> out     = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
      const int              src_dim = derez.unpack_dimension();
      switch (src_dim) {
        case 1: {
          const Point<1>         bounds = derez.unpack_point<1>();
          const AccessorRO<T, 1> in =
              derez.unpack_accessor_RO<T, 1>(regions[1], Rect<1>(Point<1>::ZEROES(), bounds - Point<1>::ONES()));
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                out[x][y][z] = in[z % bounds[0]];
          break;
        }
        case 2: {
          const Point<2>         bounds = derez.unpack_point<2>();
          const AccessorRO<T, 2> in =
              derez.unpack_accessor_RO<T, 2>(regions[1], Rect<2>(Point<2>::ZEROES(), bounds - Point<2>::ONES()));
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                out[x][y][z] = in[y % bounds[0]][z % bounds[1]];
          break;
        }
        case 3: {
          const Point<3>         bounds = derez.unpack_point<3>();
          const AccessorRO<T, 3> in =
              derez.unpack_accessor_RO<T, 3>(regions[1], Rect<3>(Point<3>::ZEROES(), bounds - Point<3>::ONES()));
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                out[x][y][z] = in[x % bounds[0]][y % bounds[1]][z % bounds[2]];
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
#endif

INSTANTIATE_ALL_TASKS(TileTask, static_cast<int>(NumPyOpCode::NUMPY_TILE) * NUMPY_TYPE_OFFSET + NUMPY_NORMAL_VARIANT_OFFSET)

}    // namespace numpy
}    // namespace legate

namespace    // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { REGISTER_ALL_TASKS(legate::numpy::TileTask) }
}    // namespace
