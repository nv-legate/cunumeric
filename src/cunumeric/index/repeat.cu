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
  count_repeat_kernel(const size_t volume,
                      Output out,
                      const AccessorRO<int64_t, DIM> repeats,
                      const Pitches<DIM - 1> pitches,
                      const Point<DIM> origin,
                      const size_t iters,
                      Buffer<int64_t> offsets)
{
  int64_t value = 0;
  for (size_t idx = 0; idx < iters; idx++) {
    const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < volume) {
      auto point      = pitches.unflatten(offset, origin);
      auto val        = repeats[point];
      offsets[offset] = val;
      SumReduction<int64_t>::fold<true>(value, val);
    }
  }
  // Every thread in the thread block must participate in the exchange to get correct results
  reduce_output(out, value);
}

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  repeat_kernel(Buffer<VAL> out,
                const AccessorRO<VAL, DIM> in,
                int64_t repeats,
                const int32_t axis,
                const Rect<DIM> rect,
                const Pitches<DIM - 1> pitches,
                const int volume)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  int64_t i = idx / repeats;
  auto p    = pitches.unflatten(i, rect.lo);
  out[idx]  = in[p];
}

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  repeat_kernel(Buffer<VAL> out,
                const AccessorRO<VAL, DIM> in,
                const AccessorRO<int64_t, DIM> repeats,
                Buffer<int64_t> offsets,
                const int32_t axis,
                const Rect<DIM> rect,
                const Pitches<DIM - 1> pitches,
                const int volume)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto p         = pitches.unflatten(idx, rect.lo);
  size_t out_idx = offsets[idx];
  for (size_t r = 0; r < repeats[p]; r++) {
    out[out_idx] = in[p];
    ++out_idx;
  }
}

template <LegateTypeCode CODE, int DIM>
struct RepeatImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  size_t operator()(Buffer<VAL>& out,
                    const AccessorRO<VAL, DIM>& in,
                    const int64_t repeats,
                    const int32_t axis,
                    const Pitches<DIM - 1>& pitches,
                    const Rect<DIM>& rect) const
  {
    const size_t volume = rect.volume();
    size_t size         = volume * repeats;
    const size_t blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    out = create_buffer<VAL>(size, Memory::Kind::GPU_FB_MEM);

    repeat_kernel<VAL, DIM>
      <<<blocks, THREADS_PER_BLOCK>>>(out, in, repeats, axis, rect, pitches, size);
    return size;
  }

  size_t operator()(Buffer<VAL>& out,
                    const AccessorRO<VAL, DIM>& in,
                    const AccessorRO<int64_t, DIM>& repeats,
                    const int32_t axis,
                    const Pitches<DIM - 1>& pitches,
                    const Rect<DIM>& rect) const
  {
    const size_t volume = rect.volume();

    auto stream = get_cached_stream();

    // compute offsets
    Buffer<int64_t> offsets = create_buffer<int64_t>(volume, Memory::Kind::GPU_FB_MEM);
    DeferredReduction<SumReduction<int64_t>> size;
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t shmem_size   = THREADS_PER_BLOCK / 32 * sizeof(int64_t);

    if (blocks > MAX_REDUCTION_CTAS) {
      const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
      count_repeat_kernel<<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, size, repeats, pitches, rect.lo, iters, offsets);
    } else {
      count_repeat_kernel<<<blocks, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, size, repeats, pitches, rect.lo, 1, offsets);
    }

    cudaStreamSynchronize(stream);

    auto p_offsets = offsets.ptr(0);

    exclusive_sum(p_offsets, volume, stream);

    auto out_size = size.read();

    out = create_buffer<VAL>(out_size, Memory::Kind::GPU_FB_MEM);

    repeat_kernel<VAL, DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      out, in, repeats, offsets, axis, rect, pitches, volume);
    return out_size;
  }

  static void exclusive_sum(int64_t* offsets, size_t volume, cudaStream_t stream)
  {
    thrust::exclusive_scan(thrust::cuda::par.on(stream), offsets, offsets + volume, offsets);
  }
};

/*static*/ void RepeatTask::gpu_variant(TaskContext& context)
{
  repeat_template<VariantKind::GPU>(context);
}
}  // namespace cunumeric
