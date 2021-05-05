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

#include "binary_op.h"
#include "core.h"
#include "deserializer.h"
#include "dispatch.h"
#include "point_task.h"

namespace legate {
namespace numpy {

using namespace Legion;

void deserialize(Deserializer &ctx, BinaryOpCode &code)
{
  int32_t value;
  deserialize(ctx, value);
  code = static_cast<BinaryOpCode>(value);
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

    auto out = out_rf.write_accessor<RES, DIM>();
    auto in1 = in1_rf.read_accessor<ARG, DIM>();
    auto in2 = in2_rf.read_accessor<ARG, DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in1.accessor.is_dense_row_major(rect) &&
                 in2.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    if (volume == 0) return;

    OP func{};
    if (dense) {
      auto outptr = out.ptr(rect);
      auto in1ptr = in1.ptr(rect);
      auto in2ptr = in2.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) outptr[idx] = func(in1ptr[idx], in2ptr[idx]);
    } else {
      CPULoop<DIM>::binary_loop(func, out, in1, in2, rect);
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

/*static*/ void BinaryOpTask::cpu_variant(const Task *task,
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

  op_dispatch(op_code, BinaryOpDispatch{}, shape, out, in1, in2);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { BinaryOpTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
