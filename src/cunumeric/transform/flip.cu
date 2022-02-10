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

#include "cunumeric/transform/flip.h"
#include "cunumeric/transform/flip_template.inl"

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <typename WriteAcc, typename ReadAcc, typename Pitches, typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  flip_kernel(const size_t volume,
              WriteAcc out,
              ReadAcc in,
              Pitches pitches,
              Rect rect,
              DeferredBuffer<int32_t, 1> axes,
              const uint32_t num_axes)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto p = pitches.unflatten(idx, rect.lo);
  auto q = p;
  for (uint32_t idx = 0; idx < num_axes; ++idx) q[axes[idx]] = rect.hi[axes[idx]] - q[axes[idx]];
  out[p] = in[q];
}

template <LegateTypeCode CODE, int32_t DIM>
struct FlipImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, DIM> in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  legate::Span<const int32_t> axes) const

  {
    const size_t volume = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto num_axes       = axes.size();
    DeferredBuffer<int32_t, 1> gpu_axes(Memory::Kind::Z_COPY_MEM, Rect<1>(0, num_axes - 1));
    for (uint32_t idx = 0; idx < num_axes; ++idx) gpu_axes[idx] = axes[idx];
    flip_kernel<<<blocks, THREADS_PER_BLOCK>>>(volume, out, in, pitches, rect, gpu_axes, num_axes);
  }
};

/*static*/ void FlipTask::gpu_variant(TaskContext& context)
{
  flip_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
