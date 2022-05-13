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

#include "cunumeric/index/advanced_indexing.h"
#include "cunumeric/index/advanced_indexing_template.inl"
#include "cunumeric/cuda_help.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

namespace cunumeric {

using namespace Legion;

template <typename Output, typename Pitches, typename Point, int32_t DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  count_nonzero_kernel(size_t volume,
                       Output out,
                       AccessorRO<bool, DIM> index,
                       Pitches pitches,
                       Point origin,
                       size_t iters,
                       const size_t skip_size)
{
  size_t value = 0;
  for (size_t idx = 0; idx < iters; idx++) {
    size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    offset        = offset * skip_size;
    if (offset < volume) {
      auto point = pitches.unflatten(offset, origin);
      auto val   = static_cast<size_t>(index[point]);
      SumReduction<size_t>::fold<true>(value, val);
    }
  }
  // Every thread in the thread block must participate in the exchange to get correct results
  reduce_output(out, value);
}

template <typename Pitches, typename Point, int DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  compute_offsets_kernel(size_t volume,
                         AccessorRO<bool, DIM> index,
                         Pitches pitches,
                         Point origin,
                         size_t iters,
                         Buffer<int64_t> offsets,
                         const size_t skip_size)
{
  for (size_t idx = 0; idx < iters; idx++) {
    const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < volume) {
      auto point = pitches.unflatten(offset, origin);
      if (index[point] and ((idx != 0 and idx % skip_size == 0) or (skip_size == 1))) {
        offsets[offset] = 1;

      } else {
        offsets[offset] = 0;
      }
    }
  }
}

template <typename VAL, int DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  advanced_indexing_kernel(size_t volume,
                           AccessorRO<VAL, DIM> in,
                           AccessorRO<bool, DIM> index,
                           Buffer<VAL, DIM> out,
                           Pitches<DIM - 1> pitches,
                           Point<DIM> origin,
                           Buffer<int64_t> offsets,
                           const size_t skip_size,
                           const size_t key_dim)
{
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= volume) return;
  auto point = pitches.unflatten(tid, origin);
  if (index[point] == true) {
    Point<DIM> out_p;
    out_p[0] = offsets[tid];
    for (int i = key_dim; i < DIM; i++) { out_p[i - key_dim + 1] = point[i]; }
    out[out_p] = in[point];
  }
}

template <typename VAL, int DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  advanced_indexing_kernel(size_t volume,
                           AccessorRO<VAL, DIM> in,
                           AccessorRO<bool, DIM> index,
                           Buffer<Point<DIM>, DIM> out,
                           Pitches<DIM - 1> pitches,
                           Point<DIM> origin,
                           Buffer<int64_t> offsets,
                           const size_t skip_size,
                           const size_t key_dim)
{
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= volume) return;
  auto point = pitches.unflatten(tid, origin);
  if (index[point] == true) {
    Point<DIM> out_p;
    out_p[0] = offsets[tid];
    for (int i = key_dim; i < DIM; i++) { out_p[i - key_dim + 1] = point[i]; }
    out[out_p] = point;
  }
}

template <LegateTypeCode CODE, int DIM, typename OUT_TYPE>
struct AdvancedIndexingImplBody<VariantKind::GPU, CODE, DIM, OUT_TYPE> {
  using VAL = legate_type_of<CODE>;

  int64_t compute_size(const AccessorRO<bool, DIM>& in,
                       const Pitches<DIM - 1>& pitches,
                       const Rect<DIM>& rect,
                       const size_t volume,
                       cudaStream_t stream,
                       Buffer<int64_t>& offsets,
                       const size_t skip_size) const
  {
    DeferredReduction<SumReduction<size_t>> size;

    const size_t blocks1 = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const size_t volume2 = (volume + skip_size) / skip_size;
    const size_t blocks2 = (volume2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t shmem_size    = THREADS_PER_BLOCK / 32 * sizeof(int64_t);

    if (blocks2 >= MAX_REDUCTION_CTAS) {
      const size_t iters = (blocks2 + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
      count_nonzero_kernel<<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, size, in, pitches, rect.lo, iters, skip_size);
    } else
      count_nonzero_kernel<<<blocks2, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, size, in, pitches, rect.lo, 1, skip_size);

    compute_offsets_kernel<<<blocks1, THREADS_PER_BLOCK, shmem_size, stream>>>(
      volume, in, pitches, rect.lo, 1, offsets, skip_size);

    cudaStreamSynchronize(stream);

    auto off_ptr = offsets.ptr(0);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), off_ptr, off_ptr + volume, off_ptr);

    return size.read();
  }

  size_t operator()(Array& out_arr,
                    const AccessorRO<VAL, DIM>& input,
                    const AccessorRO<bool, DIM>& index,
                    const Pitches<DIM - 1>& pitches,
                    const Rect<DIM>& rect,
                    const size_t key_dim) const
  {
    size_t size         = 0;
    const size_t volume = rect.volume();
    auto stream         = get_cached_stream();
    auto offsets        = create_buffer<int64_t, 1>(volume, Memory::Kind::GPU_FB_MEM);
    Point<DIM> extends;

    size_t skip_size = 1;
    for (int i = key_dim; i < DIM; i++) {
      auto diff                = 1 + rect.hi[i] - rect.lo[i];
      extends[i - key_dim + 1] = diff;
      if (diff != 0) skip_size *= diff;
    }

    size       = compute_size(index, pitches, rect, volume, stream, offsets, skip_size);
    extends[0] = size;

    auto out = out_arr.create_output_buffer<OUT_TYPE, DIM>(extends, Memory::Kind::GPU_FB_MEM);

    // populate output
    if (size > 0) {
      const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      advanced_indexing_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        volume, input, index, out, pitches, rect.lo, offsets, skip_size, key_dim);
    }
    CHECK_CUDA_STREAM(stream);
    return size;
  }
};

/*static*/ void AdvancedIndexingTask::gpu_variant(TaskContext& context)
{
  advanced_indexing_template<VariantKind::GPU>(context);
}
}  // namespace cunumeric
