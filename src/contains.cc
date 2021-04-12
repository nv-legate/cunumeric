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
#include "proj.h"
#ifdef LEGATE_USE_OPENMP
#  include <omp.h>
#endif

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
/*static*/ bool ContainsTask<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                             Runtime* runtime) {
  assert(task->futures.size() == 1);
  const T            value = task->futures[0].template get_result<T>();
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        if (in[x] == value) return true;
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          if (in[x][y] == value) return true;
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
            if (in[x][y][z] == value) return true;
      break;
    }
    default:
      assert(false);
  }
  // If we make it here then we don't have the value
  return false;
}

#ifdef LEGATE_USE_OPENMP
template<typename T>
/*static*/ bool ContainsTask<T>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                             Runtime* runtime) {
  assert(task->futures.size() == 1);
  const T            value = task->futures[0].template get_result<T>();
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim    = derez.unpack_dimension();
  bool               result = false;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        if (in[x] == value) result = true;
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[0], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          if (in[x][y] == value) result = true;
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[0], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
            if (in[x][y][z] == value) result = true;
      break;
    }
    default:
      assert(false);
  }
  return result;
}
#endif

INSTANTIATE_ALL_TASKS(ContainsTask,
                      static_cast<int>(NumPyOpCode::NUMPY_CONTAINS) * NUMPY_TYPE_OFFSET + NUMPY_REDUCTION_VARIANT_OFFSET)

}    // namespace numpy
}    // namespace legate

namespace    // unnammed
{
static void __attribute__((constructor)) register_tasks(void) { REGISTER_ALL_TASKS_WITH_BOOL_RETURN(legate::numpy::ContainsTask) }
}    // namespace
