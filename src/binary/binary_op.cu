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

#include "binary/binary_op.h"
#include "core.h"
#include "dispatch.h"
#include "point_task.h"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace legate {
namespace numpy {

using namespace Legion;

namespace gpu {

template <typename Function, typename ARG, typename RES>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  dense_kernel(size_t volume, Function func, RES *out, const ARG *in1, const ARG *in2)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  out[idx] = func(in1[idx], in2[idx]);
}

template <typename Function, typename ReadAcc, typename WriteAcc, typename Pitches, typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) generic_kernel(
  size_t volume, Function func, WriteAcc out, ReadAcc in1, ReadAcc in2, Pitches pitches, Rect rect)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto point = pitches.unflatten(idx, rect.lo);
  out[point] = func(in1[point], in2[point]);
}

template <BinaryOpCode OP_CODE>
struct BinaryOpImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<BinaryOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(Shape &shape, RegionField &out_rf, RegionField &in1_rf, RegionField &in2_rf)
  {
    using OP  = BinaryOp<OP_CODE, CODE>;
    using ARG = legate_type_of<CODE>;
    using RES = std::result_of_t<OP(ARG, ARG)>;

    auto rect = shape.to_rect<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out = out_rf.write_accessor<RES, DIM>();
    auto in1 = in1_rf.read_accessor<ARG, DIM>();
    auto in2 = in2_rf.read_accessor<ARG, DIM>();

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in1.accessor.is_dense_row_major(rect) &&
                 in2.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    OP func{};
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dense) {
      auto outptr = out.ptr(rect);
      auto in1ptr = in1.ptr(rect);
      auto in2ptr = in2.ptr(rect);
      dense_kernel<<<blocks, THREADS_PER_BLOCK>>>(volume, func, outptr, in1ptr, in2ptr);
    } else {
      generic_kernel<<<blocks, THREADS_PER_BLOCK>>>(volume, func, out, in1, in2, pitches, rect);
    }
  }

  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<!BinaryOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(Shape &shape, RegionField &out_rf, RegionField &in1_rf, RegionField &in2_rf)
  {
    assert(false);
  }
};

struct BinaryOpDispatch {
  template <BinaryOpCode OP_CODE>
  void operator()(Shape &shape, RegionField &out, RegionField &in1, RegionField &in2)
  {
    double_dispatch(in1.dim(), in1.code(), BinaryOpImpl<OP_CODE>{}, shape, out, in1, in2);
  }
};

}  // namespace gpu

/*static*/ void BinaryOpTask::gpu_variant(const Task *task,
                                          const std::vector<PhysicalRegion> &regions,
                                          Context context,
                                          Runtime *runtime)
{
  Deserializer ctx(task, regions);

  BinaryOpCode op_code;
  Shape shape;
  RegionField out;
  RegionField in1;
  RegionField in2;

  deserialize(ctx, op_code);
  deserialize(ctx, shape);
  deserialize(ctx, out);
  deserialize(ctx, in1);
  deserialize(ctx, in2);

  op_dispatch(op_code, gpu::BinaryOpDispatch{}, shape, out, in1, in2);
}

}  // namespace numpy
}  // namespace legate
