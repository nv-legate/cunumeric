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

#include "cunumeric/random/rand.h"
#include "cunumeric/random/rand_template.inl"

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;

template <typename WriteAcc, typename Rng, int32_t DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) rand_kernel(
  size_t volume, WriteAcc out, Rng rng, Point<DIM> strides, Pitches<DIM - 1> pitches, Point<DIM> lo)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto point    = pitches.unflatten(idx, lo);
  size_t offset = 0;
  for (size_t dim = 0; dim < DIM; ++dim) offset += point[dim] * strides[dim];
  out[point] = rng(HI_BITS(offset), LO_BITS(offset));
}

template <typename RNG, typename VAL, int32_t DIM>
struct RandImplBody<VariantKind::GPU, RNG, VAL, DIM> {
  void operator()(AccessorWO<VAL, DIM> out,
                  const RNG& rng,
                  const Point<DIM>& strides,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect) const
  {
    size_t volume       = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    rand_kernel<<<blocks, THREADS_PER_BLOCK>>>(volume, out, rng, strides, pitches, rect.lo);
  }
};

/*static*/ void RandTask::gpu_variant(TaskContext& context)
{
  rand_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
