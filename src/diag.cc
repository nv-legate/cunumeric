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

#include "diag.h"
#include "proj.h"

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
template<typename TASK>
/*static*/ void DiagTask<T>::set_layout_constraints(LegateVariant variant, TaskLayoutConstraintSet& layout_constraints) {
  // Don't put constraints on the first region requirement as it
  // could either be a reduction instance or a normal instance
  // depending on whether we are doing an index space launch or not
  for (int idx = 1; idx < TASK::REGIONS; idx++)
    layout_constraints.add_layout_constraint(idx, Core::get_soa_layout());
}

template<typename T>
/*static*/ void DiagTask<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                         Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          k      = derez.unpack_32bit_int();
  const Rect<2>      rect2d = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
  // Always get the 1-D rect from the logical region since
  // we can't easily predict its shape, the shape could be 2-D if we're
  // going to be reducing into a reduction buffer
  const Domain dom = runtime->get_index_space_domain(task->regions[0].region.get_index_space());
  assert((dom.get_dim() == 1) || (dom.get_dim() == 2));
  const Rect<1> rect1d = (dom.get_dim() == 1) ? Rect<1>(dom) : extract_rect1d(dom);
  if (rect1d.empty()) return;
  // Solve for the start
  // y = x + k
  // x >= rect2d.lo[0]
  const Point<2> start1(rect2d.lo[0], rect2d.lo[0] + k);
  // y >= rect2d.lo[1]
  const Point<2> start2(rect2d.lo[1] - k, rect2d.lo[1]);
  // See if our rect intersects with the diagonal
  if (!rect2d.contains(start1) && !rect2d.contains(start2)) {
    // Still have to write identity into our rectangle for diagonal out case
    if (task->regions[0].privilege != READ_ONLY) {
      const int              collapse_dim   = derez.unpack_dimension();
      const int              collapse_index = derez.unpack_dimension();
      const AccessorWO<T, 1> out =
          (collapse_dim >= 0) ? derez.unpack_accessor_WO<T, 1>(regions[0], rect1d, collapse_dim, task->index_point[collapse_index])
                              : derez.unpack_accessor_WO<T, 1>(regions[0], rect1d);
      // Initialize the output
      for (coord_t x = rect1d.lo; x <= rect1d.hi; x++)
        out[x] = SumReduction<T>::identity;
    }
    return;
  }
  // Pick whichever one fits in our rect
  const Point<2> start = rect2d.contains(start1) ? start1 : start2;
  // Now do the same thing for the end
  // x <= rect2d.hi[0]
  const Point<2> stop1(rect2d.hi[0], rect2d.hi[0] + k);
  // y <= rect2d.hi[1]
  const Point<2> stop2(rect2d.hi[1] - k, rect2d.hi[1]);
  assert(rect2d.contains(stop1) || rect2d.contains(stop2));
  const Point<2> stop = rect2d.contains(stop1) ? stop1 : stop2;
  // Walk the path from the stop to the start
  const coord_t distance = (stop[0] - start[0]) + 1;
  // Should be the same along both dimensions
  assert(distance == ((stop[1] - start[1]) + 1));
  if (task->regions[0].privilege == READ_ONLY) {
    // diagonal in
    // Should also be less than or the same as the volume of our 1d rect
    assert(distance <= (coord_t)rect1d.volume());
    const AccessorRO<T, 1> in  = derez.unpack_accessor_RO<T, 1>(regions[0], rect1d);
    const AccessorRW<T, 2> out = derez.unpack_accessor_RW<T, 2>(regions[1], rect2d);
    for (coord_t idx = 0; idx < distance; idx++)
      out[start[0] + idx][start[1] + idx] = in[rect1d.lo + idx];
  } else {
    // diagonal out
    const int              collapse_dim   = derez.unpack_dimension();
    const int              collapse_index = derez.unpack_dimension();
    const AccessorWO<T, 1> out =
        (collapse_dim >= 0) ? derez.unpack_accessor_WO<T, 1>(regions[0], rect1d, collapse_dim, task->index_point[collapse_index])
                            : derez.unpack_accessor_WO<T, 1>(regions[0], rect1d);
    const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], rect2d);
    // Initialize the output
    for (coord_t x = rect1d.lo; x <= rect1d.hi; x++)
      out[x] = SumReduction<T>::identity;
    // Figure out which dimension we align on
    if ((rect1d.lo == rect2d.lo[0]) && (rect1d.hi == rect2d.hi[0])) {
      for (coord_t idx = 0; idx < distance; idx++)
        out[start[0] + idx] = in[start[0] + idx][start[1] + idx];
    } else {
      assert((rect1d.lo == rect2d.lo[1]) && (rect1d.hi == rect2d.hi[1]));
      for (coord_t idx = 0; idx < distance; idx++)
        out[start[1] + idx] = in[start[0] + idx][start[1] + idx];
    }
  }
}

#ifdef LEGATE_USE_OPENMP
template<typename T>
/*static*/ void DiagTask<T>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                         Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          k      = derez.unpack_32bit_int();
  const Rect<2>      rect2d = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
  // Always get the 1-D rect from the logical region since
  // we can't easily predict its shape, the shape could be 2-D if we're
  // going to be reducing into a reduction buffer
  const Domain dom = runtime->get_index_space_domain(task->regions[0].region.get_index_space());
  assert((dom.get_dim() == 1) || (dom.get_dim() == 2));
  const Rect<1> rect1d = (dom.get_dim() == 1) ? Rect<1>(dom) : extract_rect1d(dom);
  if (rect1d.empty()) return;
  // Solve for the start
  // y = x + k
  // x >= rect2d.lo[0]
  const Point<2> start1(rect2d.lo[0], rect2d.lo[0] + k);
  // y >= rect2d.lo[1]
  const Point<2> start2(rect2d.lo[1] - k, rect2d.lo[1]);
  // See if our rect intersects with the diagonal
  if (!rect2d.contains(start1) && !rect2d.contains(start2)) {
    // Still have to write identity into our rectangle for diagonal out case
    if (task->regions[0].privilege != READ_ONLY) {
      const int              collapse_dim   = derez.unpack_dimension();
      const int              collapse_index = derez.unpack_dimension();
      const AccessorWO<T, 1> out =
          (collapse_dim >= 0) ? derez.unpack_accessor_WO<T, 1>(regions[0], rect1d, collapse_dim, task->index_point[collapse_index])
                              : derez.unpack_accessor_WO<T, 1>(regions[0], rect1d);
// Initialize the output
#  pragma omp parallel for
      for (coord_t x = rect1d.lo; x <= rect1d.hi; x++)
        out[x] = SumReduction<T>::identity;
    }
    return;
  }
  // Pick whichever one fits in our rect
  const Point<2> start = rect2d.contains(start1) ? start1 : start2;
  // Now do the same thing for the end
  // x <= rect2d.hi[0]
  const Point<2> stop1(rect2d.hi[0], rect2d.hi[0] + k);
  // y <= rect2d.hi[1]
  const Point<2> stop2(rect2d.hi[1] - k, rect2d.hi[1]);
  assert(rect2d.contains(stop1) || rect2d.contains(stop2));
  const Point<2> stop = rect2d.contains(stop1) ? stop1 : stop2;
  // Walk the path from the stop to the start
  const coord_t distance = (stop[0] - start[0]) + 1;
  // Should be the same along both dimensions
  assert(distance == ((stop[1] - start[1]) + 1));
  if (task->regions[0].privilege == READ_ONLY) {
    // diagonal in
    // Should also be less than or the same as the volume of our 1d rect
    assert(distance <= (coord_t)rect1d.volume());
    const AccessorRO<T, 1> in  = derez.unpack_accessor_RO<T, 1>(regions[0], rect1d);
    const AccessorRW<T, 2> out = derez.unpack_accessor_RW<T, 2>(regions[1], rect2d);
#  pragma omp parallel for
    for (coord_t idx = 0; idx < distance; idx++)
      out[start[0] + idx][start[1] + idx] = in[rect1d.lo + idx];
  } else {
    // diagonal out
    const int              collapse_dim   = derez.unpack_dimension();
    const int              collapse_index = derez.unpack_dimension();
    const AccessorWO<T, 1> out =
        (collapse_dim >= 0) ? derez.unpack_accessor_WO<T, 1>(regions[0], rect1d, collapse_dim, task->index_point[collapse_index])
                            : derez.unpack_accessor_WO<T, 1>(regions[0], rect1d);
    const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], rect2d);
// Initialize the output
#  pragma omp parallel for
    for (coord_t x = rect1d.lo; x <= rect1d.hi; x++)
      out[x] = SumReduction<T>::identity;
    // Figure out which dimension we align on
    if ((rect1d.lo == rect2d.lo[0]) && (rect1d.hi == rect2d.hi[0])) {
#  pragma omp parallel for
      for (coord_t idx = 0; idx < distance; idx++)
        out[start[0] + idx] = in[start[0] + idx][start[1] + idx];
    } else {
      assert((rect1d.lo == rect2d.lo[1]) && (rect1d.hi == rect2d.hi[1]));
#  pragma omp parallel for
      for (coord_t idx = 0; idx < distance; idx++)
        out[start[1] + idx] = in[start[0] + idx][start[1] + idx];
    }
  }
}
#endif    // LEGATE_USE_OPENMP

INSTANTIATE_ALL_TASKS(DiagTask, static_cast<int>(NumPyOpCode::NUMPY_DIAG) * NUMPY_TYPE_OFFSET + NUMPY_NORMAL_VARIANT_OFFSET)
}    // namespace numpy
}    // namespace legate

namespace    // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { REGISTER_ALL_TASKS(legate::numpy::DiagTask) }
}    // namespace
