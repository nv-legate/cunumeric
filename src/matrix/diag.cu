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
                const AccessorRO<VAL, 1> in,
                const coord_t distance,
                const Point<2> start_out,
                const coord_t start_in)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= distance) return;
  out[start_out[0] + idx][start_out[1] + idx] = in[start_in + idx];
}

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  diag_extract(const AccessorWO<VAL, 1> out,
               const AccessorRO<VAL, 2> in,
               const coord_t distance,
               const coord_t start_out,
               const Point<2> start_in)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= distance) return;
  out[start_out + idx] = in[start_in[0] + idx][start_in[1] + idx];
}

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  diag_extract(const AccessorRD<SumReduction<VAL>, true, 1> out,
               const AccessorRO<VAL, 2> in,
               const coord_t distance,
               const coord_t start_out,
               const Point<2> start_in)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= distance) return;
  out.reduce(start_out + idx, in[start_in[0] + idx][start_in[1] + idx]);
}

template <LegateTypeCode CODE>
struct DiagImplBody<VariantKind::GPU, CODE> {
  using VAL = legate_type_of<CODE>;

  void operator()(const AccessorWO<VAL, 2> &out,
                  const AccessorRO<VAL, 1> &in,
                  coord_t distance,
                  const Point<2> &start_out,
                  coord_t start_in) const
  {
    const size_t blocks = (distance + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    diag_populate<VAL><<<blocks, THREADS_PER_BLOCK>>>(out, in, distance, start_out, start_in);
  }

  void operator()(const AccessorWO<VAL, 1> &out,
                  const AccessorRO<VAL, 2> &in,
                  coord_t distance,
                  coord_t start_out,
                  const Point<2> &start_in) const
  {
    const size_t blocks = (distance + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    diag_extract<VAL><<<blocks, THREADS_PER_BLOCK>>>(out, in, distance, start_out, start_in);
  }

  void operator()(const AccessorRD<SumReduction<VAL>, true, 1> &out,
                  const AccessorRO<VAL, 2> &in,
                  coord_t distance,
                  coord_t start_out,
                  const Point<2> &start_in) const
  {
    const size_t blocks = (distance + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    diag_extract<VAL><<<blocks, THREADS_PER_BLOCK>>>(out, in, distance, start_out, start_in);
  }
};

/*static*/ void DiagTask::gpu_variant(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context context,
                                      Runtime *runtime)
{
  diag_template<VariantKind::GPU>(task, regions, context, runtime);
}

}  // namespace numpy
}  // namespace legate
