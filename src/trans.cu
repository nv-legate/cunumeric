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
#include "proj.h"
#include "trans.h"

#define TILE_DIM 32
#define BLOCK_ROWS 8

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
__global__ void __launch_bounds__((TILE_DIM * BLOCK_ROWS), MIN_CTAS_PER_SM)
    legate_transpose_2d(const AccessorWO<T, 2> out, const AccessorRO<T, 2> in, const Point<2> lo_in, const Point<2> hi_in,
                        const Point<2> lo_out, const Point<2> hi_out) {
  __shared__ T tile[TILE_DIM][TILE_DIM + 1 /*avoid bank conflicts*/];

  // These are reversed here for coalescing
  coord_t x = blockIdx.y * TILE_DIM + threadIdx.y;
  coord_t y = blockIdx.x * TILE_DIM + threadIdx.x;

  // Check to see if we hit our y-bounds, if so we can just mask off those threads
  if ((lo_in[1] + y) <= hi_in[1]) {
    // Check to see if we're going to hit our x-bounds while striding
    if ((lo_in[0] + (blockIdx.y + 1) * TILE_DIM - 1) <= hi_in[0]) {
// No overflow case
#pragma unroll
      for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        tile[threadIdx.y + i][threadIdx.x] = in[lo_in + Point<2>(x + i, y)];
    } else {
// Overflow case
#pragma unroll
      for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        if ((lo_in[0] + x + i) <= hi_in[0]) tile[threadIdx.y + i][threadIdx.x] = in[lo_in + Point<2>(x + i, y)];
    }
  }
  // Make sure all the data is in shared memory
  __syncthreads();

  // Transpose the coordinates
  x = blockIdx.x * TILE_DIM + threadIdx.y;
  y = blockIdx.y * TILE_DIM + threadIdx.x;

  // Check to see if we hit our y-bounds, if so we can just mask off those threads
  if ((lo_out[1] + y) <= hi_out[1]) {
    // Check to see if we're going to hit our x-bounds while striding
    if ((lo_out[0] + (blockIdx.x + 1) * TILE_DIM - 1) <= hi_out[0]) {
// No overflow case
#pragma unroll
      for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        out[lo_out + Point<2>(x + i, y)] = tile[threadIdx.x][threadIdx.y + i];
    } else {
// Overflow case
#pragma unroll
      for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        if ((lo_out[0] + x + i) <= hi_out[0]) out[lo_out + Point<2>(x + i, y)] = tile[threadIdx.x][threadIdx.y + i];
    }
  }
}

template<typename T>
/*static*/ void TransTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                          Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  switch (dim) {
    case 2: {
      const Rect<2> out_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (out_rect.empty()) break;
      const AccessorWO<T, 2> out = derez.unpack_accessor_WO<T, 2>(regions[0], out_rect);
      // We know what the output shape has to be based on the input shape
      const Rect<2>          in_rect(Point<2>(out_rect.lo[1], out_rect.lo[0]), Point<2>(out_rect.hi[1], out_rect.hi[0]));
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], in_rect);
      const coord_t          m  = (in_rect.hi[0] - in_rect.lo[0]) + 1;
      const coord_t          n  = (in_rect.hi[1] - in_rect.lo[1]) + 1;
      const dim3             blocks((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM, 1);
      const dim3             threads(TILE_DIM, BLOCK_ROWS, 1);
      legate_transpose_2d<T><<<blocks, threads>>>(out, in, in_rect.lo, in_rect.hi, out_rect.lo, out_rect.hi);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

INSTANTIATE_TASK_VARIANT(TransTask, gpu_variant)

}    // namespace numpy
}    // namespace legate
