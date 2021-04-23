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

#include "eye.h"
#include "proj.h"

using namespace Legion;

namespace legate {
namespace numpy {

template <typename T>
/*static*/ void EyeTask<T>::cpu_variant(const Task* task,
                                        const std::vector<PhysicalRegion>& regions,
                                        Context ctx,
                                        Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  // We know this is 2-D
  const Rect<2> rect         = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
  const AccessorRW<T, 2> out = derez.unpack_accessor_RW<T, 2>(regions[0], rect);
  const int k                = derez.unpack_32bit_int();
  // Solve for the start
  // y = x + k
  // x >= rect.lo[0]
  const Point<2> start1(rect.lo[0], rect.lo[0] + k);
  // y >= rect.lo[1]
  const Point<2> start2(rect.lo[1] - k, rect.lo[1]);
  // If we don't have a start point then there's nothing for us to do
  if (!rect.contains(start1) && !rect.contains(start2)) return;
  // Pick whichever one fits in our rect
  const Point<2> start = rect.contains(start1) ? start1 : start2;
  // Now do the same thing for the end
  // x <= rect.hi[0]
  const Point<2> stop1(rect.hi[0], rect.hi[0] + k);
  // y <= rect.hi[1]
  const Point<2> stop2(rect.hi[1] - k, rect.hi[1]);
  assert(rect.contains(stop1) || rect.contains(stop2));
  const Point<2> stop = rect.contains(stop1) ? stop1 : stop2;
  // Walk the path from the stop to the start
  const coord_t distance = (stop[0] - start[0]) + 1;
  // Should be the same along both dimensions
  assert(distance == ((stop[1] - start[1]) + 1));
  for (coord_t idx = 0; idx < distance; idx++) out[start[0] + idx][start[1] + idx] = T{1};
}

#ifdef LEGATE_USE_OPENMP
template <typename T>
/*static*/ void EyeTask<T>::omp_variant(const Task* task,
                                        const std::vector<PhysicalRegion>& regions,
                                        Context ctx,
                                        Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  // We know this is 2-D
  const Rect<2> rect         = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
  const AccessorRW<T, 2> out = derez.unpack_accessor_RW<T, 2>(regions[0], rect);
  const int k                = derez.unpack_32bit_int();
  // Solve for the start
  // y = x + k
  // x >= rect.lo[0]
  const Point<2> start1(rect.lo[0], rect.lo[0] + k);
  // y >= rect.lo[1]
  const Point<2> start2(rect.lo[1] - k, rect.lo[1]);
  // If we don't have a start point then there's nothing for us to do
  if (!rect.contains(start1) && !rect.contains(start2)) return;
  // Pick whichever one fits in our rect
  const Point<2> start = rect.contains(start1) ? start1 : start2;
  // Now do the same thing for the end
  // x <= rect.hi[0]
  const Point<2> stop1(rect.hi[0], rect.hi[0] + k);
  // y <= rect.hi[1]
  const Point<2> stop2(rect.hi[1] - k, rect.hi[1]);
  assert(rect.contains(stop1) || rect.contains(stop2));
  const Point<2> stop = rect.contains(stop1) ? stop1 : stop2;
  // Walk the path from the stop to the start
  const coord_t distance = (stop[0] - start[0]) + 1;
  // Should be the same along both dimensions
  assert(distance == ((stop[1] - start[1]) + 1));
#pragma omp parallel for
  for (coord_t idx = 0; idx < distance; idx++) out[start[0] + idx][start[1] + idx] = 1;
}
#endif

INSTANTIATE_ALL_TASKS(EyeTask,
                      static_cast<int>(NumPyOpCode::NUMPY_EYE) * NUMPY_TYPE_OFFSET +
                        NUMPY_NORMAL_VARIANT_OFFSET)

}  // namespace numpy
}  // namespace legate

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  REGISTER_ALL_TASKS(legate::numpy::EyeTask)
}
}  // namespace
