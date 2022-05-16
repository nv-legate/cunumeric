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
                       Buffer<int64_t> offsets,
                       AccessorRO<bool, DIM> index,
                       Pitches pitches,
                       Point origin,
                       size_t iters,
                       const size_t skip_size,
                       const size_t key_dim)
{
  size_t value = 0;
  for (size_t i = 0; i < iters; i++) {
    size_t idx = (i * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx > volume) return;
    auto point = pitches.unflatten(idx, origin);
    size_t val = 0;
    //    if (index[point] and (((idx+1)%skip_size == 0) || (idx==0 &&key_dim==DIM))){ offsets[idx]
    //    = 1;
    if (index[point] and ((idx + 1) % skip_size == 0)) {
      offsets[idx] = 1;
      val          = 1;
    } else {
      offsets[idx] = 0;
    }
    SumReduction<size_t>::fold<true>(value, val);
  }
  // Every thread in the thread block must participate in the exchange to get correct results
  reduce_output(out, value);
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
  if (tid > volume) return;
  auto point = pitches.unflatten(tid, origin);
  if (index[point] == true) {
    Point<DIM> out_p;
    for (int i = 0; i < key_dim - 1; i++) { out_p[i] = 0; }
    out_p[key_dim - 1] = offsets[tid];
    for (int i = key_dim; i < DIM; i++) { out_p[i] = point[i]; }
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
    for (int i = 0; i < key_dim - 1; i++) { out_p[i] = 0; }
    out_p[key_dim - 1] = offsets[tid];
    for (int i = key_dim; i < DIM; i++) { out_p[i] = point[i]; }
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
                       const size_t skip_size,
                       const size_t key_dim) const
  {
    DeferredReduction<SumReduction<size_t>> size;

    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    size_t shmem_size = THREADS_PER_BLOCK / 32 * sizeof(int64_t);

    if (blocks >= MAX_REDUCTION_CTAS) {
      const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
      count_nonzero_kernel<<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, size, offsets, in, pitches, rect.lo, iters, skip_size, key_dim);
    } else
      count_nonzero_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        volume, size, offsets, in, pitches, rect.lo, 1, skip_size, key_dim);

    cudaStreamSynchronize(stream);

    auto off_ptr = offsets.ptr(0);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), off_ptr, off_ptr + volume, off_ptr);

    return size.read();
  }

  void operator()(Array& out_arr,
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
      auto diff  = 1 + rect.hi[i] - rect.lo[i];
      extends[i] = diff;
      if (diff != 0) skip_size *= diff;
    }
    for (int i = 0; i < key_dim - 1; i++) extends[i] = 1;

    size = compute_size(index, pitches, rect, volume, stream, offsets, skip_size, key_dim);
    extends[key_dim - 1] = size;

    auto out = out_arr.create_output_buffer<OUT_TYPE, DIM>(extends, true);

    // populate output
    if (size > 0) {
      const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      advanced_indexing_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        volume, input, index, out, pitches, rect.lo, offsets, skip_size, key_dim);
    }
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void AdvancedIndexingTask::gpu_variant(TaskContext& context)
{
  advanced_indexing_template<VariantKind::GPU>(context);
}
}  // namespace cunumeric
