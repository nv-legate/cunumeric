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
                       Buffer<int64_t> offsets)
{
  int64_t value = 0;
  for (size_t idx = 0; idx < iters; idx++) {
    const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < volume) {
      auto point      = pitches.unflatten(offset, origin);
      auto val        = static_cast<int64_t>(index[point]);
      offsets[offset] = val;
      SumReduction<int64_t>::fold<true>(value, val);
    }
  }
  // Every thread in the thread block must participate in the exchange to get correct results
  reduce_output(out, value);
}

template <typename VAL, int DIM1, int DIM2>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  advanced_indexing_kernel(size_t volume,
                           AccessorRO<VAL, DIM1> in,
                           AccessorRO<bool, DIM2> index,
                           Buffer<VAL> out,
                           Pitches<DIM1 - 1> pitches_input,
                           Point<DIM1> origin_input,
                           Pitches<DIM2 - 1> pitches_index,
                           Point<DIM2> origin_index,
                           Buffer<int64_t> offsets)
{
  // FIXME works only when DIM1==DIM2
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= volume) return;
  auto point       = pitches_index.unflatten(tid, origin_index);
  auto point_input = pitches_input.unflatten(tid, origin_input);
  if (index[point] == true) {
    int64_t offset = offsets[tid];
    out[offset]    = in[point_input];
  }
}

template <typename VAL, int DIM1, int DIM2>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  advanced_indexing_kernel(size_t volume,
                           AccessorRO<VAL, DIM1> in,
                           AccessorRO<bool, DIM2> index,
                           Buffer<Point<DIM1>> out,
                           Pitches<DIM1 - 1> pitches_input,
                           Point<DIM1> origin_input,
                           Pitches<DIM2 - 1> pitches_index,
                           Point<DIM2> origin_index,
                           Buffer<int64_t> offsets)
{
  // FIXME works only when DIM1==DIM2
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= volume) return;
  auto point       = pitches_index.unflatten(tid, origin_index);
  auto point_input = pitches_input.unflatten(tid, origin_input);
  if (index[point] == true) {
    int64_t offset = offsets[tid];
    out[offset]    = point_input;
  }
}

template <LegateTypeCode CODE, int DIM1, int DIM2, bool IS_SET>
struct AdvancedIndexingImplBody<VariantKind::GPU, CODE, DIM1, DIM2, IS_SET> {
  using VAL = legate_type_of<CODE>;

  int64_t compute_size(const AccessorRO<bool, DIM2>& in,
                       const Pitches<DIM2 - 1>& pitches,
                       const Rect<DIM2>& rect,
                       const size_t volume,
                       cudaStream_t stream,
                       Buffer<int64_t>& offsets) const
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

    auto off_ptr = offsets.ptr(0);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), off_ptr, off_ptr + volume, off_ptr);

    return size.read();
  }

  template <typename OUT_TYPE>
  size_t operator()(Buffer<OUT_TYPE>& out,
                    const AccessorRO<VAL, DIM1>& input,
                    const AccessorRO<bool, DIM2>& index,
                    const Pitches<DIM1 - 1>& pitches_input,
                    const Rect<DIM1>& rect_input,
                    const Pitches<DIM2 - 1>& pitches_index,
                    const Rect<DIM2>& rect_index) const
  {
#ifdef CUNUMERIC_DEBUG
    // in this case shapes for input and index arrays  should be the same
    assert(rect_input == rect_index);
#endif
    int64_t size          = 0;
    const bool* index_ptr = index.ptr(rect_index);
    const size_t volume   = rect_index.volume();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto offsets = create_buffer<int64_t>(volume, Memory::Kind::GPU_FB_MEM);
    size         = compute_size(index, pitches_index, rect_index, volume, stream, offsets);

    out = create_buffer<OUT_TYPE>(size, Memory::Kind::GPU_FB_MEM);
    // populate output
    if (size > 0) {
      const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      advanced_indexing_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(volume,
                                                                         input,
                                                                         index,
                                                                         out,
                                                                         pitches_input,
                                                                         rect_input.lo,
                                                                         pitches_index,
                                                                         rect_index.lo,
                                                                         offsets);
    }
    return size;
  }
};

/*static*/ void AdvancedIndexingTask::gpu_variant(TaskContext& context)
{
  advanced_indexing_template<VariantKind::GPU>(context);
}
}  // namespace cunumeric
