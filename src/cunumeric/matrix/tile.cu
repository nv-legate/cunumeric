/* Copyright 2021-2022 NVIDIA Corporation
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

#include "cunumeric/matrix/tile.h"
#include "cunumeric/matrix/tile_template.inl"

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;

template <typename VAL, int32_t OUT_DIM, int32_t IN_DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  tile_kernel(const Rect<OUT_DIM> out_rect,
              const Pitches<OUT_DIM - 1> out_pitches,
              const size_t out_volume,
              const Point<IN_DIM> in_strides,
              const AccessorWO<VAL, OUT_DIM> out,
              const AccessorRO<VAL, IN_DIM> in)
{
  const size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_idx >= out_volume) return;

  const auto out_point = out_pitches.unflatten(out_idx, out_rect.lo);
  const auto in_point  = get_tile_point(out_point, in_strides);
  out[out_point]       = in[in_point];
}

template <typename VAL, int32_t OUT_DIM, int32_t IN_DIM>
struct TileImplBody<VariantKind::GPU, VAL, OUT_DIM, IN_DIM> {
  void operator()(const Rect<OUT_DIM>& out_rect,
                  const Pitches<OUT_DIM - 1>& out_pitches,
                  size_t out_volume,
                  const Point<IN_DIM>& in_strides,
                  const AccessorWO<VAL, OUT_DIM>& out,
                  const AccessorRO<VAL, IN_DIM>& in) const
  {
    const size_t blocks = (out_volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto stream         = get_cached_stream();
    tile_kernel<VAL, OUT_DIM, IN_DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      out_rect, out_pitches, out_volume, in_strides, out, in);
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void TileTask::gpu_variant(TaskContext& context)
{
  tile_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
