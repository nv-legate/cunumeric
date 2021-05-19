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

#include "broadcast_binary_red.h"
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
  dense_kernel(size_t volume, Function func, RES out, const ARG *in1, ARG in2)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  if (func(in1[idx], in2)) out << false;
}

template <typename Function,
          typename ReadAcc,
          typename ARG,
          typename RES,
          typename Pitches,
          typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) generic_kernel(
  size_t volume, Function func, RES out, ReadAcc in1, ARG in2, Pitches pitches, Rect rect)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto point = pitches.unflatten(idx, rect.lo);
  if (func(in1[point], in2)) out << false;
}

template <BinaryOpCode OP_CODE,
          LegateTypeCode TYPE,
          typename Acc,
          typename T,
          typename Rect,
          typename Pitches>
static inline UntypedScalar binary_red_loop(BinaryOp<OP_CODE, TYPE> func,
                                            const Acc &in1,
                                            const T &in2,
                                            const Rect &rect,
                                            const Pitches &pitches,
                                            bool dense)
{
  size_t volume = rect.volume();

  const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  DeferredReduction<ProdReduction<bool>> result;
  if (dense) {
    auto in1ptr = in1.ptr(rect);
    dense_kernel<<<blocks, THREADS_PER_BLOCK>>>(volume, func, result, in1ptr, in2);
  } else {
    generic_kernel<<<blocks, THREADS_PER_BLOCK>>>(volume, func, result, in1, in2, pitches, rect);
  }

  return UntypedScalar(result.read());
}

template <BinaryOpCode OP_CODE>
struct BroadcastBinaryRedImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<BinaryOp<OP_CODE, CODE>::valid> * = nullptr>
  UntypedScalar operator()(Shape &shape,
                           RegionField &in1_rf,
                           UntypedScalar &in2_scalar,
                           std::vector<UntypedScalar> &args)
  {
    using OP  = BinaryOp<OP_CODE, CODE>;
    using ARG = legate_type_of<CODE>;

    auto rect = shape.to_rect<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return UntypedScalar(true);

    auto in1 = in1_rf.read_accessor<ARG, DIM>();
    auto in2 = in2_scalar.value<ARG>();

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = in1.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    OP func(args);
    return binary_red_loop(func, in1, in2, rect, pitches, dense);
  }

  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<!BinaryOp<OP_CODE, CODE>::valid> * = nullptr>
  UntypedScalar operator()(Shape &shape,
                           RegionField &in1_rf,
                           UntypedScalar &in2_scalar,
                           std::vector<UntypedScalar> &args)
  {
    assert(false);
    return UntypedScalar();
  }
};

struct BroadcastBinaryRedDispatch {
  template <BinaryOpCode OP_CODE>
  UntypedScalar operator()(Shape &shape,
                           RegionField &in1,
                           UntypedScalar &in2,
                           std::vector<UntypedScalar> &args)
  {
    return double_dispatch(
      in1.dim(), in1.code(), BroadcastBinaryRedImpl<OP_CODE>{}, shape, in1, in2, args);
  }
};

}  // namespace gpu

/*static*/ UntypedScalar BroadcastBinaryRedTask::gpu_variant(
  const Task *task, const std::vector<PhysicalRegion> &regions, Context context, Runtime *runtime)
{
  Deserializer ctx(task, regions);

  BinaryOpCode op_code;
  Shape shape;
  RegionField in1;
  UntypedScalar in2;
  std::vector<UntypedScalar> args;

  deserialize(ctx, op_code);
  deserialize(ctx, shape);
  deserialize(ctx, in1);
  deserialize(ctx, in2);
  deserialize(ctx, args);

  return reduce_op_dispatch(op_code, gpu::BroadcastBinaryRedDispatch{}, shape, in1, in2, args);
}

}  // namespace numpy
}  // namespace legate
