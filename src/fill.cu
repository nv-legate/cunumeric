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

#include "fill.h"
#include "core.h"
#include "dispatch.h"
#include "point_task.h"
#include "scalar.h"

namespace legate {
namespace numpy {

using namespace Legion;

namespace gpu {

template <typename ARG, typename RES>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  dense_kernel(size_t volume, RES *out, ARG fill_value)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  out[idx] = fill_value;
}

template <typename ARG, typename WriteAcc, typename Pitches, typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  generic_kernel(size_t volume, WriteAcc out, ARG fill_value, Pitches pitches, Rect rect)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto point = pitches.unflatten(idx, rect.lo);
  out[point] = fill_value;
}

struct FillImpl {
  template <LegateTypeCode CODE, int DIM>
  void operator()(Shape &shape, RegionField &out_rf, UntypedScalar &fill_value_scalar)
  {
    using VAL = legate_type_of<CODE>;

    auto rect = shape.to_rect<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out        = out_rf.write_accessor<VAL, DIM>();
    auto fill_value = fill_value_scalar.value<VAL>();

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dense) {
      auto outptr = out.ptr(rect);
      dense_kernel<<<blocks, THREADS_PER_BLOCK>>>(volume, outptr, fill_value);
    } else
      generic_kernel<<<blocks, THREADS_PER_BLOCK>>>(volume, out, fill_value, pitches, rect);
  }
};

}  // namespace gpu

/*static*/ void FillTask::gpu_variant(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context context,
                                      Runtime *runtime)
{
  Deserializer ctx(task, regions);

  Shape shape;
  RegionField out;
  UntypedScalar fill_value;

  deserialize(ctx, shape);
  deserialize(ctx, out);
  deserialize(ctx, fill_value);

  double_dispatch(out.dim(), out.code(), gpu::FillImpl{}, shape, out, fill_value);
}

}  // namespace numpy
}  // namespace legate
