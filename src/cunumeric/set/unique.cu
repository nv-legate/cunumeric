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

#include "cunumeric/set/unique.h"
#include "cunumeric/set/unique_template.inl"

#include "cunumeric/cuda_help.h"

#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  copy_into_buffer(VAL* out,
                   const AccessorRO<VAL, DIM> accessor,
                   const Point<DIM> lo,
                   const Pitches<DIM - 1> pitches,
                   const size_t volume)
{
  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= volume) return;
  auto point  = pitches.unflatten(offset, lo);
  out[offset] = accessor[lo + point];
}

template <LegateTypeCode CODE, int32_t DIM>
struct UniqueImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  std::pair<Buffer<VAL>, size_t> operator()(const AccessorRO<VAL, DIM>& in,
                                            const Pitches<DIM - 1>& pitches,
                                            const Rect<DIM>& rect,
                                            const size_t volume)
  {
    auto stream = get_cached_stream();

    // Make a copy of the input as we're going to sort it
    auto temp = create_buffer<VAL>(volume);
    VAL* ptr  = temp.ptr(0);
    if (in.accessor.is_dense_arbitrary(rect)) {
      auto* src = in.ptr(rect.lo);
      cudaMemcpyAsync(ptr, src, sizeof(VAL) * volume, cudaMemcpyDeviceToDevice, stream);
    } else {
      const size_t num_blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      copy_into_buffer<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        ptr, in, rect.lo, pitches, volume);
    }

    // Find unique values
    thrust::sort(thrust::cuda::par.on(stream), ptr, ptr + volume);
    auto* end = thrust::unique(thrust::cuda::par.on(stream), ptr, ptr + volume);

    // Finally we pack the result
    size_t size = end - ptr;
    auto result = create_buffer<VAL>(size);
    cudaMemcpyAsync(result.ptr(0), ptr, sizeof(VAL) * size, cudaMemcpyDeviceToDevice, stream);
    return std::make_pair(result, size);
  }
};

/*static*/ void UniqueTask::gpu_variant(TaskContext& context)
{
  unique_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
