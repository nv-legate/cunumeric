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

#include "broadcast_binary_op.h"
#include "binary_op_util.h"
#include "core.h"
#include "dispatch.h"
#include "point_task.h"
#include "scalar.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <BinaryOpCode OP_CODE>
struct BinaryOpImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<BinaryOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(Shape &shape,
                  RegionField &out_rf,
                  RegionField &in1_rf,
                  UntypedScalar &in2_scalar,
                  bool future_on_rhs)
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
    auto in2 = in2_scalar.value<ARG>();

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in1.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    OP func{};
    if (dense) {
      auto outptr = out.ptr(rect);
      auto in1ptr = in1.ptr(rect);
      if (future_on_rhs)
        for (size_t idx = 0; idx < volume; ++idx) outptr[idx] = func(in1ptr[idx], in2);
      else
        for (size_t idx = 0; idx < volume; ++idx) outptr[idx] = func(in2, in1ptr[idx]);
    } else {
      if (future_on_rhs)
        CPULoop<DIM>::binary_loop(func, out, in1, Scalar<ARG, DIM>(in2), rect);
      else
        CPULoop<DIM>::binary_loop(func, out, Scalar<ARG, DIM>(in2), in1, rect);
    }
  }

  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<!BinaryOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(Shape &shape,
                  RegionField &out_rf,
                  RegionField &in1_rf,
                  UntypedScalar &in2_scalar,
                  bool future_on_rhs)
  {
    assert(false);
  }
};

struct BinaryOpDispatch {
  template <BinaryOpCode OP_CODE>
  void operator()(
    Shape &shape, RegionField &out, RegionField &in1, UntypedScalar &in2, bool future_on_rhs)
  {
    double_dispatch(
      in1.dim(), in1.code(), BinaryOpImpl<OP_CODE>{}, shape, out, in1, in2, future_on_rhs);
  }
};

/*static*/ void BroadcastBinaryOpTask::cpu_variant(const Task *task,
                                                   const std::vector<PhysicalRegion> &regions,
                                                   Context context,
                                                   Runtime *runtime)
{
  Deserializer ctx(task, regions);

  BinaryOpCode op_code;
  Shape shape;
  RegionField out;
  RegionField in1;
  UntypedScalar in2;
  bool future_on_rhs;

  deserialize(ctx, op_code);
  deserialize(ctx, shape);
  deserialize(ctx, out);
  deserialize(ctx, in1);
  deserialize(ctx, in2);
  deserialize(ctx, future_on_rhs);

  op_dispatch(op_code, BinaryOpDispatch{}, shape, out, in1, in2, future_on_rhs);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  BroadcastBinaryOpTask::register_variants();
}
}  // namespace

}  // namespace numpy
}  // namespace legate
