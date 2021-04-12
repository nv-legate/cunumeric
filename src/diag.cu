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

#include "cuda_help.h"
#include "diag.h"
#include "fill.cuh"
#include "proj.h"

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_diag_in(const AccessorRW<T, 2> out, const AccessorRO<T, 1> in, const Point<2> start_out, const Point<1> start_in,
                   const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  out[start_out[0] + offset][start_out[1] + offset] = in[start_in + offset];
}

template<typename T, bool FIRST>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_diag_out(const AccessorWO<T, 1> out, const AccessorRO<T, 2> in, const Point<2> start, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  if (FIRST)
    out[start[0] + offset] = in[start[0] + offset][start[1] + offset];
  else
    out[start[1] + offset] = in[start[0] + offset][start[1] + offset];
}

template<typename T>
/*static*/ void DiagTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
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
      const size_t volume = rect1d.volume();
      const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      legate_fill_1d<T><<<blocks, THREADS_PER_BLOCK>>>(out, SumReduction<T>::identity, rect1d.lo, volume);
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
  // Figure out how many CTAs to launch
  const size_t blocks = (distance + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  if (task->regions[0].privilege == READ_ONLY) {
    // diagonal in
    // Should also be less than or the same as the volume of our 1d rect
    assert(distance <= (coord_t)rect1d.volume());
    const AccessorRO<T, 1> in  = derez.unpack_accessor_RO<T, 1>(regions[0], rect1d);
    const AccessorRW<T, 2> out = derez.unpack_accessor_RW<T, 2>(regions[1], rect2d);
    legate_diag_in<T><<<blocks, THREADS_PER_BLOCK>>>(out, in, start, rect1d.lo, distance);
  } else {
    // diagonal out
    const int              collapse_dim   = derez.unpack_dimension();
    const int              collapse_index = derez.unpack_dimension();
    const AccessorWO<T, 1> out =
        (collapse_dim >= 0) ? derez.unpack_accessor_WO<T, 1>(regions[0], rect1d, collapse_dim, task->index_point[collapse_index])
                            : derez.unpack_accessor_WO<T, 1>(regions[0], rect1d);
    const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], rect2d);
    // Initialize the output
    const size_t volume = rect1d.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    legate_fill_1d<T><<<blocks, THREADS_PER_BLOCK>>>(out, SumReduction<T>::identity, rect1d.lo, volume);
    // Figure out which dimension we align on
    const size_t dist_blocks = (distance + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if ((rect1d.lo == rect2d.lo[0]) && (rect1d.hi == rect2d.hi[0])) {
      legate_diag_out<T, true><<<dist_blocks, THREADS_PER_BLOCK>>>(out, in, start, distance);
    } else {
      assert((rect1d.lo == rect2d.lo[1]) && (rect1d.hi == rect2d.hi[1]));
      legate_diag_out<T, false><<<dist_blocks, THREADS_PER_BLOCK>>>(out, in, start, distance);
    }
  }
}

INSTANTIATE_TASK_VARIANT(DiagTask, gpu_variant)

}    // namespace numpy
}    // namespace legate
