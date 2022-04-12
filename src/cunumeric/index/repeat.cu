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

#include "cunumeric/index/repeat.h"
#include "cunumeric/index/repeat_template.inl"
#include "cunumeric/cuda_help.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

namespace cunumeric {

using namespace Legion;

template <typename Output, int DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  count_repeat_kernel(const int64_t extent,
                      Output sum,
                      const AccessorRO<int64_t, DIM> repeats,
                      const Point<DIM> origin,
                      const int32_t axis,
                      const size_t iters,
                      Buffer<int64_t> offsets)
{
  int64_t value = 0;
  for (size_t idx = 0; idx < iters; idx++) {
    const int64_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < extent) {
      auto p = origin;
      p[axis] += offset;
      auto val        = repeats[p];
      offsets[offset] = val;
      SumReduction<int64_t>::fold<true>(value, val);
    }
  }
  // Every thread in the thread block must participate in the exchange to get correct results
  reduce_output(sum, value);
}

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  repeat_kernel(Buffer<VAL, DIM> out,
                const AccessorRO<VAL, DIM> in,
                int64_t repeats,
                const int32_t axis,
                const Point<DIM> in_lo,
                const Pitches<DIM - 1> pitches,
                const size_t volume)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto out_p = pitches.unflatten(idx, Point<DIM>::ZEROES());
  auto in_p  = out_p;
  in_p[axis] /= repeats;
  in_p += in_lo;
  out[out_p] = in[in_p];
}

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  repeat_kernel(Buffer<VAL, DIM> out,
                const AccessorRO<VAL, DIM> in,
                const AccessorRO<int64_t, DIM> repeats,
                Buffer<int64_t> offsets,
                const int32_t axis,
                const Point<DIM> in_lo,
                const Pitches<DIM - 1> pitches,
                const int volume)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto in_p  = pitches.unflatten(idx, in_lo);
  auto out_p = in_p - in_lo;

  int64_t off_start = offsets[in_p[axis] - in_lo[axis]];
  int64_t off_end   = off_start + repeats[in_p];

  auto in_v = in[in_p];
  for (int64_t out_idx = off_start; out_idx < off_end; ++out_idx) {
    out_p[axis] = out_idx;
    out[out_p]  = in_v;
  }
}

template <LegateTypeCode CODE, int DIM>
struct RepeatImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(Array& out_array,
                  const AccessorRO<VAL, DIM>& in,
                  const int64_t repeats,
                  const int32_t axis,
                  const Rect<DIM>& in_rect) const
  {
    Point<DIM> extents = in_rect.hi - in_rect.lo + Point<DIM>::ONES();
    extents[axis] *= repeats;

    auto out = out_array.create_output_buffer<VAL, DIM>(extents, true);

    Rect<DIM> out_rect(Point<DIM>::ZEROES(), extents - Point<DIM>::ONES());
    Pitches<DIM - 1> pitches;

    auto out_volume   = pitches.flatten(out_rect);
    const auto blocks = (out_volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    repeat_kernel<VAL, DIM>
      <<<blocks, THREADS_PER_BLOCK>>>(out, in, repeats, axis, in_rect.lo, pitches, out_volume);
  }

  void operator()(Array& out_array,
                  const AccessorRO<VAL, DIM>& in,
                  const AccessorRO<int64_t, DIM>& repeats,
                  const int32_t axis,
                  const Rect<DIM>& in_rect) const
  {
    auto stream = get_cached_stream();

    Pitches<DIM - 1> pitches;
    const auto volume = pitches.flatten(in_rect);

    // Compute offsets
    int64_t extent = in_rect.hi[axis] - in_rect.lo[axis] + 1;
    auto offsets   = create_buffer<int64_t>(Point<1>(extent), Memory::Kind::Z_COPY_MEM);

    DeferredReduction<SumReduction<int64_t>> sum;
    const size_t blocks_count = (extent + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const size_t shmem_size   = THREADS_PER_BLOCK / 32 * sizeof(int64_t);

    if (blocks_count > MAX_REDUCTION_CTAS) {
      const size_t iters = (blocks_count + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
      count_repeat_kernel<<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK, shmem_size, stream>>>(
        extent, sum, repeats, in_rect.lo, axis, iters, offsets);
    } else {
      count_repeat_kernel<<<blocks_count, THREADS_PER_BLOCK, shmem_size, stream>>>(
        extent, sum, repeats, in_rect.lo, axis, 1, offsets);
    }

    cudaStreamSynchronize(stream);

    Point<DIM> out_extents = in_rect.hi - in_rect.lo + Point<DIM>::ONES();
    out_extents[axis]      = sum.read();

    auto out = out_array.create_output_buffer<VAL, DIM>(out_extents, true);

    auto p_offsets = offsets.ptr(0);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), p_offsets, p_offsets + extent, p_offsets);

    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    repeat_kernel<VAL, DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      out, in, repeats, offsets, axis, in_rect.lo, pitches, volume);
  }
};

/*static*/ void RepeatTask::gpu_variant(TaskContext& context)
{
  repeat_template<VariantKind::GPU>(context);
}
}  // namespace cunumeric
