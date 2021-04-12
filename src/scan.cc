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

#include "scan.h"
#include "proj.h"
#include <numeric>

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
/*static*/ void InclusiveScanTask<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                  Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  Rect<1>            rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  if (rect.empty()) return;
  AccessorRW<T, 1> inout  = derez.unpack_accessor_RW<T, 1>(regions[0], rect);
  T*               ptr    = inout.ptr(rect);
  const size_t     volume = rect.volume();
  std::partial_sum(ptr, ptr + volume, ptr);
}

#ifdef LEGATE_USE_OPENMP
template<typename T>
/*static*/ void InclusiveScanTask<T>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                  Runtime* runtime) {
  // TBD: Need OpenMP implementation
  cpu_variant(task, regions, ctx, runtime);
}
#endif    // LEGATE_USE_OPENMP

INSTANTIATE_ALL_TASKS(InclusiveScanTask, static_cast<int>(NumPyOpCode::NUMPY_INCLUSIVE_SCAN) * NUMPY_TYPE_OFFSET)

}    // namespace numpy
}    // namespace legate

namespace    // unnammed
{
static void __attribute__((constructor)) register_tasks(void) { REGISTER_ALL_TASKS(legate::numpy::InclusiveScanTask) }
}    // namespace
