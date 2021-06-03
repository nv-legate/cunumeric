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

template <typename T, int DIM>
__global__ void __launch_bounds__(1, 1) legate_read_value(DeferredValue<T> value,
                                                          const AccessorRO<T, DIM> accessor,
                                                          const Point<DIM> point)
{
  value = accessor[point];
}

template <typename T>
/*static*/ DeferredValue<T> ReadItemTask<T>::gpu_variant(
  const Legion::Task* task,
  const std::vector<Legion::PhysicalRegion>& regions,
  Legion::Context ctx,
  Legion::Runtime* runtime)
{
  DeferredValue<T> result((T)0);
  LegateDeserializer derez(task->args, task->arglen);
  const int dim = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const AccessorRO<T, 1> acc = derez.unpack_accessor_RO<T, 1>(regions[0]);
      const Point<1> p           = derez.unpack_point<1>();
      legate_read_value<T, 1><<<1, 1>>>(result, acc, p);
      break;
    }
    case 2: {
      const AccessorRO<T, 2> acc = derez.unpack_accessor_RO<T, 2>(regions[0]);
      const Point<2> p           = derez.unpack_point<2>();
      legate_read_value<T, 2><<<1, 1>>>(result, acc, p);
      break;
    }
    case 3: {
      const AccessorRO<T, 3> acc = derez.unpack_accessor_RO<T, 3>(regions[0]);
      const Point<3> p           = derez.unpack_point<3>();
      legate_read_value<T, 3><<<1, 1>>>(result, acc, p);
      break;
    }
    default: assert(false);
  }
  return result;
}

INSTANTIATE_DEFERRED_VALUE_TASK_VARIANTS(ReadItemTask, gpu_variant)

template <typename T, int DIM>
__global__ void __launch_bounds__(1, 1)
  legate_write_value(const AccessorRW<T, DIM> accessor, const Point<DIM> point, const T value)
{
  accessor[point] = value;
}

template <typename T>
/*static*/ void WriteItemTask<T>::gpu_variant(const Legion::Task* task,
                                              const std::vector<Legion::PhysicalRegion>& regions,
                                              Legion::Context ctx,
                                              Legion::Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int dim = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const AccessorRW<T, 1> acc = derez.unpack_accessor_RW<T, 1>(regions[0]);
      const Point<1> p           = derez.unpack_point<1>();
      legate_write_value<T, 1><<<1, 1>>>(acc, p, derez.unpack_value<T>());
      break;
    }
    case 2: {
      const AccessorRW<T, 2> acc = derez.unpack_accessor_RW<T, 2>(regions[0]);
      const Point<2> p           = derez.unpack_point<2>();
      legate_write_value<T, 2><<<1, 1>>>(acc, p, derez.unpack_value<T>());
      break;
    }
    case 3: {
      const AccessorRW<T, 3> acc = derez.unpack_accessor_RW<T, 3>(regions[0]);
      const Point<3> p           = derez.unpack_point<3>();
      legate_write_value<T, 3><<<1, 1>>>(acc, p, derez.unpack_value<T>());
      break;
    }
    default: assert(false);
  }
}

INSTANTIATE_TASK_VARIANT(WriteItemTask, gpu_variant)

}  // namespace numpy
}  // namespace legate
