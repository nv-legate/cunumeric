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

#include "search/nonzero.h"
#include "search/nonzero_template.inl"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

namespace legate {
namespace numpy {

using namespace Legion;

template <typename Output, typename Pitches, typename Point, typename VAL, int32_t DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  count_nonzero_kernel(size_t volume,
                       Output out,
                       AccessorRO<VAL, DIM> in,
                       Pitches pitches,
                       Point origin,
                       size_t iters,
                       DeferredBuffer<int64_t, 1> offsets)
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
                 DeferredBuffer<int64_t, 1> offsets,
                 DeferredBuffer<int64_t *, 1> p_results)
{
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= volume) return;

  auto point = pitches.unflatten(tid, origin);
  if (in[point] != VAL(0)) {
    auto offset = offsets[tid];
    for (int32_t dim = 0; dim < DIM; ++dim) p_results[dim][offset] = point[dim];
  }
}

static void exclusive_sum(int64_t *offsets, size_t volume, cudaStream_t stream)
{
  thrust::exclusive_scan(thrust::cuda::par.on(stream), offsets, offsets + volume, offsets);
}

template <LegateTypeCode CODE, int32_t DIM>
struct NonzeroImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  size_t compute_offsets(const AccessorRO<VAL, DIM> &in,
                         const Pitches<DIM - 1> &pitches,
                         const Rect<DIM> &rect,
                         const size_t volume,
                         DeferredBuffer<int64_t, 1> &offsets,
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

    return size.read();
  }

  void populate_nonzeros(const AccessorRO<VAL, DIM> &in,
                         const Pitches<DIM - 1> &pitches,
                         const Rect<DIM> &rect,
                         const size_t volume,
                         std::vector<DeferredBuffer<int64_t, 1>> &results,
                         DeferredBuffer<int64_t, 1> &offsets,
                         cudaStream_t stream)
  {
    auto ndims = static_cast<int32_t>(results.size());
    DeferredBuffer<int64_t *, 1> p_results(Rect<1>(0, ndims - 1), Memory::Kind::Z_COPY_MEM);
    for (int32_t dim = 0; dim < ndims; ++dim) p_results[dim] = results[dim].ptr(0);

    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    nonzero_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      volume, in, pitches, rect.lo, offsets, p_results);
  }

  size_t operator()(const AccessorRO<VAL, DIM> &in,
                    const Pitches<DIM - 1> &pitches,
                    const Rect<DIM> &rect,
                    const size_t volume,
                    std::vector<DeferredBuffer<int64_t, 1>> &results)
  {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    DeferredBuffer<int64_t, 1> offsets(Rect<1>(0, volume - 1), Memory::Kind::GPU_FB_MEM);

    int64_t size = compute_offsets(in, pitches, rect, volume, offsets, stream);

    for (auto &result : results) {
      auto hi = std::max<int64_t>(size - 1, 0);
      result  = DeferredBuffer<int64_t, 1>(Rect<1>(0, hi), Memory::Kind::GPU_FB_MEM);
    }

    if (size > 0) populate_nonzeros(in, pitches, rect, volume, results, offsets, stream);
    // int64_t out_idx = 0;
    // for (size_t idx = 0; idx < volume; ++idx) {
    //  auto point = pitches.unflatten(idx, rect.lo);
    //  if (in[point] == VAL(0)) continue;
    //  for (int32_t dim = 0; dim < DIM; ++dim) results[dim][out_idx] = point[dim];
    //  ++out_idx;
    //}
    // assert(size == out_idx);

    return size;
  }
};

/*static*/ void NonzeroTask::gpu_variant(const Legion::Task *task,
                                         const std::vector<Legion::PhysicalRegion> &regions,
                                         Legion::Context ctx,
                                         Legion::Runtime *runtime)
{
  nonzero_template<VariantKind::GPU>(task, regions, ctx, runtime);
}

}  // namespace numpy
}  // namespace legate
