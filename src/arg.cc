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
#include "proj.h"

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
/*static*/ void GetargTask<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
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
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        out[x] = in[x].arg;
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<int64_t, 2>   out = derez.unpack_accessor_WO<int64_t, 2>(regions[0], rect);
      const AccessorRO<Argval<T>, 2> in  = (extra_dim >= 0) ? derez.unpack_accessor_RO<Argval<T>, 2>(regions[1], rect, extra_dim, 0)
                                                           : derez.unpack_accessor_RO<Argval<T>, 2>(regions[1], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          out[x][y] = in[x][y].arg;
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<int64_t, 3>   out = derez.unpack_accessor_WO<int64_t, 3>(regions[0], rect);
      const AccessorRO<Argval<T>, 3> in  = (extra_dim >= 0) ? derez.unpack_accessor_RO<Argval<T>, 3>(regions[1], rect, extra_dim, 0)
                                                           : derez.unpack_accessor_RO<Argval<T>, 3>(regions[1], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
            out[x][y][z] = in[x][y][z].arg;
      break;
    }
    default:
      assert(false);
  }
}

#ifdef LEGATE_USE_OPENMP
template<typename T>
/*static*/ void GetargTask<T>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
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
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        out[x] = in[x].arg;
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<int64_t, 2>   out = derez.unpack_accessor_WO<int64_t, 2>(regions[0], rect);
      const AccessorRO<Argval<T>, 2> in  = (extra_dim >= 0) ? derez.unpack_accessor_RO<Argval<T>, 2>(regions[1], rect, extra_dim, 0)
                                                           : derez.unpack_accessor_RO<Argval<T>, 2>(regions[1], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          out[x][y] = in[x][y].arg;
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<int64_t, 3>   out = derez.unpack_accessor_WO<int64_t, 3>(regions[0], rect);
      const AccessorRO<Argval<T>, 3> in  = (extra_dim >= 0) ? derez.unpack_accessor_RO<Argval<T>, 3>(regions[1], rect, extra_dim, 0)
                                                           : derez.unpack_accessor_RO<Argval<T>, 3>(regions[1], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
            out[x][y][z] = in[x][y][z].arg;
      break;
    }
    default:
      assert(false);
  }
}
#endif

template<typename T>
/*static*/ int64_t GetargScalar<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                Runtime* runtime) {
  assert(task->futures.size() == 1);
  const Argval<T> value = task->futures[0].get_result<Argval<T>>();
  return value.arg;
}

INSTANTIATE_ALL_TASKS(GetargTask, static_cast<int>(NumPyOpCode::NUMPY_GETARG) * NUMPY_TYPE_OFFSET + NUMPY_NORMAL_VARIANT_OFFSET)
INSTANTIATE_ALL_TASKS(GetargScalar, static_cast<int>(NumPyOpCode::NUMPY_GETARG) * NUMPY_TYPE_OFFSET + NUMPY_SCALAR_VARIANT_OFFSET)

}    // namespace numpy
}    // namespace legate

namespace {    // unnamed
static void __attribute__((constructor)) register_tasks(void) {
  REGISTER_ALL_TASKS(legate::numpy::GetargTask)
  REGISTER_ALL_TASKS_WITH_ARG_RETURN(legate::numpy::GetargScalar)
}
}    // namespace
