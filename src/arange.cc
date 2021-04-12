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

#include "arange.h"
#include "proj.h"

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
/*static*/ void ArangeTask<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                           Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  // We know this is 2-D
  const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  if (rect.empty()) return;
  const AccessorWO<T, 1> out = derez.unpack_accessor_WO<T, 1>(regions[0], rect);

  const T start = task->futures[0].get_result<T>();
  const T stop  = task->futures[1].get_result<T>();
  const T step  = task->futures[2].get_result<T>();

  for (PointInRectIterator<1> it(rect); it(); ++it) {
    Point<1> p = *it;
    out[p]     = (T)p[0] * step + start;
  }
}

#ifdef LEGATE_USE_OPENMP
template<typename T>
/*static*/ void ArangeTask<T>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                           Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  // We know this is 2-D
  const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  if (rect.empty()) return;
  const AccessorWO<T, 1> out = derez.unpack_accessor_WO<T, 1>(regions[0], rect);

  const T start = task->futures[0].get_result<T>();
  const T stop  = task->futures[1].get_result<T>();
  const T step  = task->futures[2].get_result<T>();

  const Point<1> lo   = rect.lo;
  const size_t   size = rect.volume();
#  pragma omp    parallel for
  for (coord_t idx = 0; idx < size; ++idx) {
    Point<1> p(lo + idx);
    out[p] = (T)p[0] * step + start;
  }
}
#endif

INSTANTIATE_ALL_TASKS(ArangeTask, static_cast<int>(NumPyOpCode::NUMPY_ARANGE) * NUMPY_TYPE_OFFSET + NUMPY_NORMAL_VARIANT_OFFSET)

}    // namespace numpy
}    // namespace legate

namespace    // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { REGISTER_ALL_TASKS(legate::numpy::ArangeTask) }
}    // namespace
