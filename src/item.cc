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

#include "item.h"

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
/*static*/ T ReadItemTask<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                          Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const AccessorRO<T, 1> acc = derez.unpack_accessor_RO<T, 1>(regions[0]);
      const Point<1>         p   = derez.unpack_point<1>();
      return acc[p];
    }
    case 2: {
      const AccessorRO<T, 2> acc = derez.unpack_accessor_RO<T, 2>(regions[0]);
      const Point<2>         p   = derez.unpack_point<2>();
      return acc[p];
    }
    case 3: {
      const AccessorRO<T, 3> acc = derez.unpack_accessor_RO<T, 3>(regions[0]);
      const Point<3>         p   = derez.unpack_point<3>();
      return acc[p];
    }
    default:
      assert(false);
  }
  return T{0};
}

template<typename T>
/*static*/ void WriteItemTask<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                              Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const AccessorRW<T, 1> acc = derez.unpack_accessor_RW<T, 1>(regions[0]);
      const Point<1>         p   = derez.unpack_point<1>();
      acc[p]                     = derez.unpack_value<T>();
      break;
    }
    case 2: {
      const AccessorRW<T, 2> acc = derez.unpack_accessor_RW<T, 2>(regions[0]);
      const Point<2>         p   = derez.unpack_point<2>();
      acc[p]                     = derez.unpack_value<T>();
      break;
    }
    case 3: {
      const AccessorRW<T, 3> acc = derez.unpack_accessor_RW<T, 3>(regions[0]);
      const Point<3>         p   = derez.unpack_point<3>();
      acc[p]                     = derez.unpack_value<T>();
      break;
    }
    default:
      assert(false);
  }
}

INSTANTIATE_ALL_TASKS(ReadItemTask, static_cast<int>(NumPyOpCode::NUMPY_READ) * NUMPY_TYPE_OFFSET)
INSTANTIATE_ALL_TASKS(WriteItemTask, static_cast<int>(NumPyOpCode::NUMPY_WRITE) * NUMPY_TYPE_OFFSET)

}    // namespace numpy
}    // namespace legate

namespace    // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  REGISTER_ALL_TASKS_WITH_RETURN(legate::numpy::ReadItemTask)
  REGISTER_ALL_TASKS(legate::numpy::WriteItemTask)
}
}    // namespace
