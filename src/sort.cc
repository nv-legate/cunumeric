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

#include "sort.h"
#include "proj.h"
#include <algorithm>
#ifdef LEGATE_USE_OPENMP
#if 0  // Legion issue 492
#include <parallel/algorithm>
#endif
#endif

using namespace Legion;

namespace legate {
namespace numpy {

template <typename T>
/*static*/ void SortTask<T>::cpu_variant(const Task* task,
                                         const std::vector<PhysicalRegion>& regions,
                                         Context ctx,
                                         Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const Rect<1> rect         = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  const AccessorRW<T, 1> out = derez.unpack_accessor_RW<T, 1>(regions[0], rect);
  if (rect.empty()) return;
  T* ptr              = out.ptr(rect);
  const size_t volume = rect.volume();
  // Call C++ STL sort algorithm
  std::sort(ptr, ptr + volume);
}

#ifdef LEGATE_USE_OPENMP
template <typename T>
/*static*/ void SortTask<T>::omp_variant(const Task* task,
                                         const std::vector<PhysicalRegion>& regions,
                                         Context ctx,
                                         Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const Rect<1> rect         = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  const AccessorRW<T, 1> out = derez.unpack_accessor_RW<T, 1>(regions[0], rect);
  if (rect.empty()) return;
  T* ptr              = out.ptr(rect);
  const size_t volume = rect.volume();
  // Call parallel sort using OpenMP
#if 0  // Legion issue 492
      __gnu_parallel::sort(ptr, ptr+volume);
#else
  std::sort(ptr, ptr + volume);
#endif
}
#endif

INSTANTIATE_ALL_TASKS(SortTask,
                      static_cast<int>(NumPyOpCode::NUMPY_SORT) * NUMPY_TYPE_OFFSET +
                        NUMPY_NORMAL_VARIANT_OFFSET)

}  // namespace numpy
}  // namespace legate

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  REGISTER_ALL_TASKS(legate::numpy::SortTask)
}
}  // namespace
