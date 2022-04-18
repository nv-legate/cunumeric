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

#include "cunumeric/search/nonzero.h"
#include "cunumeric/search/nonzero_template.inl"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;

template <typename Output, typename Pitches, typename Point, typename VAL, int32_t DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  count_nonzero_kernel(size_t volume,
                       Output out,
                       AccessorRO<VAL, DIM> in,
                       Pitches pitches,
                       Point origin,
                       size_t iters,
                       Buffer<int64_t> offsets)
{
  int64_t value = 0;
  for (size_t idx = 0; idx < iters; idx++) {
    const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < volume) {
      auto point      = pitches.unflatten(offset, origin);
      auto val        = static_cast<int64_t>(in[point] != VAL(0));
      offsets[offset] = val;
      SumReduction<int64_t>::fold<true>(value, val);
    }
  }
  // Every thread in the thread block must participate in the exchange to get correct results
  reduce_output(out, value);
}

template <typename Pitches, typename Point, typename VAL, int32_t DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  nonzero_kernel(size_t volume,
                 AccessorRO<VAL, DIM> in,
                 Pitches pitches,
                 Point origin,
                 Buffer<int64_t> offsets,
                 Buffer<int64_t*> p_results)
{
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= volume) return;

  auto point = pitches.unflatten(tid, origin);
  if (in[point] != VAL(0)) {
    auto offset = offsets[tid];
    for (int32_t dim = 0; dim < DIM; ++dim) p_results[dim][offset] = point[dim];
  }
}

static void exclusive_sum(int64_t* offsets, size_t volume, cudaStream_t stream)
{
  thrust::exclusive_scan(thrust::cuda::par.on(stream), offsets, offsets + volume, offsets);
}

template <LegateTypeCode CODE, int32_t DIM>
struct NonzeroImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  int64_t compute_offsets(const AccessorRO<VAL, DIM>& in,
                          const Pitches<DIM - 1>& pitches,
                          const Rect<DIM>& rect,
                          const size_t volume,
                          Buffer<int64_t>& offsets,
                          cudaStream_t stream)
  {
    DeferredReduction<SumReduction<int64_t>> size;

    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t shmem_size   = THREADS_PER_BLOCK / 32 * sizeof(int64_t);

    if (blocks >= MAX_REDUCTION_CTAS) {
      const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
      count_nonzero_kernel<<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, size, in, pitches, rect.lo, iters, offsets);
    } else
      count_nonzero_kernel<<<blocks, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, size, in, pitches, rect.lo, 1, offsets);

    cudaStreamSynchronize(stream);

    auto p_offsets = offsets.ptr(0);

    exclusive_sum(p_offsets, volume, stream);

    CHECK_CUDA_STREAM(stream);
    return size.read();
  }

  void populate_nonzeros(const AccessorRO<VAL, DIM>& in,
                         const Pitches<DIM - 1>& pitches,
                         const Rect<DIM>& rect,
                         const size_t volume,
                         std::vector<Buffer<int64_t>>& results,
                         Buffer<int64_t>& offsets,
                         cudaStream_t stream)
  {
    auto ndims     = static_cast<int32_t>(results.size());
    auto p_results = create_buffer<int64_t*>(ndims, Memory::Kind::Z_COPY_MEM);
    for (int32_t dim = 0; dim < ndims; ++dim) p_results[dim] = results[dim].ptr(0);

    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    nonzero_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      volume, in, pitches, rect.lo, offsets, p_results);
  }

  size_t operator()(const AccessorRO<VAL, DIM>& in,
                    const Pitches<DIM - 1>& pitches,
                    const Rect<DIM>& rect,
                    const size_t volume,
                    std::vector<Buffer<int64_t>>& results)
  {
    auto stream = get_cached_stream();

    auto offsets = create_buffer<int64_t>(volume, Memory::Kind::GPU_FB_MEM);
    auto size    = compute_offsets(in, pitches, rect, volume, offsets, stream);
    CHECK_CUDA_STREAM(stream);

    for (auto& result : results) result = create_buffer<int64_t>(size, Memory::Kind::GPU_FB_MEM);

    if (size > 0) populate_nonzeros(in, pitches, rect, volume, results, offsets, stream);
    CHECK_CUDA_STREAM(stream);

    return size;
  }
};

/*static*/ void NonzeroTask::gpu_variant(TaskContext& context)
{
  nonzero_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
