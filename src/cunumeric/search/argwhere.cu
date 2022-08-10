/* Copyright 2022 NVIDIA Corporation
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

#include "cunumeric/search/argwhere.h"
#include "cunumeric/search/argwhere_template.inl"
#include "cunumeric/search/nonzero.cuh"

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;

template <typename VAL, int32_t DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  argwhere_kernel(size_t volume,
                  AccessorRO<VAL, DIM> in,
                  Pitches<DIM - 1> pitches,
                  Point<DIM> origin,
                  Buffer<int64_t> offsets,
                  Buffer<int64_t, 2> output)
{
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= volume) return;

  auto in_p = pitches.unflatten(tid, origin);
  if (in[in_p] != VAL(0)) {
    auto offset = offsets[tid];
    for (int32_t dim = 0; dim < DIM; ++dim) output[Point<2>(offset, dim)] = in_p[dim];
  }
}

template <LegateTypeCode CODE, int DIM>
struct ArgWhereImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(Array& out_array,
                  AccessorRO<VAL, DIM> input,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  size_t volume) const
  {
    auto stream = get_cached_stream();

    auto offsets = create_buffer<int64_t>(volume, Memory::Kind::GPU_FB_MEM);
    auto size    = compute_offsets(input, pitches, rect, volume, offsets, stream);
    CHECK_CUDA_STREAM(stream);

    auto out = out_array.create_output_buffer<int64_t, 2>(Point<2>(size, DIM), true);

    if (size > 0) {
      const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      argwhere_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        volume, input, pitches, rect.lo, offsets, out);
      CHECK_CUDA_STREAM(stream);
    }
  }
};

/*static*/ void ArgWhereTask::gpu_variant(TaskContext& context)
{
  argwhere_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
