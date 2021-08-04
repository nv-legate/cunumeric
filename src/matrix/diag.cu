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

#include "matrix/diag.h"
#include "matrix/diag_template.inl"
#include "cuda_help.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  diag_populate(const AccessorWO<VAL, 2> out,
                const AccessorRO<VAL, 2> in,
                const coord_t distance,
                const Point<2> start)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= distance) return;
  Point<2> p(start[0] + idx, start[1] + idx);
  out[p] = in[p];
}

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  diag_extract(const AccessorRD<SumReduction<VAL>, true, 2> out,
               const AccessorRO<VAL, 2> in,
               const coord_t distance,
               const Point<2> start)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= distance) return;
  Point<2> p(start[0] + idx, start[1] + idx);
  auto v = in[p];
  out.reduce(p, v);
}

template <LegateTypeCode CODE>
struct DiagImplBody<VariantKind::GPU, CODE> {
  using VAL = legate_type_of<CODE>;

  void operator()(const AccessorWO<VAL, 2>& out,
                  const AccessorRO<VAL, 2>& in,
                  const Point<2>& start,
                  size_t distance) const
  {
    const size_t blocks = (distance + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    diag_populate<VAL><<<blocks, THREADS_PER_BLOCK>>>(out, in, distance, start);
  }

  void operator()(const AccessorRD<SumReduction<VAL>, true, 2>& out,
                  const AccessorRO<VAL, 2>& in,
                  const Point<2>& start,
                  size_t distance) const
  {
    const size_t blocks = (distance + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    diag_extract<VAL><<<blocks, THREADS_PER_BLOCK>>>(out, in, distance, start);
  }
};

/*static*/ void DiagTask::gpu_variant(TaskContext& context)
{
  diag_template<VariantKind::GPU>(context);
}

}  // namespace numpy
}  // namespace legate
