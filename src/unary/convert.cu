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

#include "unary/convert.h"
#include "unary/convert_util.h"
#include "core.h"
#include "deserializer.h"
#include "dispatch.h"
#include "point_task.h"

namespace legate {
namespace numpy {

using namespace Legion;

namespace gpu {

template <typename Function, typename ARG, typename RES>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  dense_kernel(size_t volume, Function func, RES *out, const ARG *in)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  out[idx] = func(in[idx]);
}

template <typename Function, typename ReadAcc, typename WriteAcc, typename Pitches, typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  generic_kernel(size_t volume, Function func, WriteAcc out, ReadAcc in, Pitches pitches, Rect rect)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto point = pitches.unflatten(idx, rect.lo);
  out[point] = func(in[point]);
}

template <LegateTypeCode SRC_TYPE>
struct ConvertImpl {
  template <LegateTypeCode DST_TYPE, int DIM, std::enable_if_t<SRC_TYPE != DST_TYPE> * = nullptr>
  void operator()(Shape &shape, RegionField &out_rf, RegionField &in_rf)
  {
    using OP  = ConvertOp<DST_TYPE, SRC_TYPE>;
    using SRC = legate_type_of<SRC_TYPE>;
    using DST = legate_type_of<DST_TYPE>;

    auto rect = shape.to_rect<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out = out_rf.write_accessor<DST, DIM>();
    auto in  = in_rf.read_accessor<SRC, DIM>();

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    OP func{};
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dense) {
      auto outptr = out.ptr(rect);
      auto inptr  = in.ptr(rect);
      dense_kernel<<<blocks, THREADS_PER_BLOCK>>>(volume, func, outptr, inptr);
    } else {
      generic_kernel<<<blocks, THREADS_PER_BLOCK>>>(volume, func, out, in, pitches, rect);
    }
  }

  template <LegateTypeCode DST_TYPE, int DIM, std::enable_if_t<SRC_TYPE == DST_TYPE> * = nullptr>
  void operator()(Shape &shape, RegionField &out_rf, RegionField &in_rf)
  {
    assert(false);
  }
};

struct SourceTypeDispatch {
  template <LegateTypeCode SRC_TYPE>
  void operator()(Shape &shape, RegionField &out, RegionField &in)
  {
    double_dispatch(out.dim(), out.code(), ConvertImpl<SRC_TYPE>{}, shape, out, in);
  }
};

}  // namespace gpu

/*static*/ void ConvertTask::gpu_variant(const Task *task,
                                         const std::vector<PhysicalRegion> &regions,
                                         Context context,
                                         Runtime *runtime)
{
  Deserializer ctx(task, regions);

  Shape shape;
  RegionField out;
  RegionField in;

  deserialize(ctx, shape);
  deserialize(ctx, out);
  deserialize(ctx, in);

  type_dispatch(in.code(), gpu::SourceTypeDispatch{}, shape, out, in);
}

}  // namespace numpy
}  // namespace legate
