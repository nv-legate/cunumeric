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

#include "binary_red.h"
#include "core.h"
#include "deserializer.h"
#include "dispatch.h"
#include "point_task.h"

namespace legate {
namespace numpy {

using namespace Legion;

namespace omp {

template <BinaryOpCode OP_CODE, LegateTypeCode TYPE, typename Acc, typename Rect, typename Pitches>
static inline UntypedScalar binary_red_loop(BinaryOp<OP_CODE, TYPE> func,
                                            const Acc &in1,
                                            const Acc &in2,
                                            const Rect &rect,
                                            const Pitches &pitches,
                                            bool dense)
{
  size_t volume = rect.volume();

  bool result = true;

  if (dense) {
    auto in1ptr = in1.ptr(rect);
    auto in2ptr = in2.ptr(rect);
#pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < volume; ++idx)
      if (!func(in1ptr[idx], in2ptr[idx])) result = false;
  } else {
#pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < volume; ++idx) {
      auto point = pitches.unflatten(idx, rect.lo);
      if (!func(in1[point], in2[point])) result = false;
    }
  }

  return UntypedScalar(result);
}

template <BinaryOpCode OP_CODE>
struct BinaryOpImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<BinaryOp<OP_CODE, CODE>::valid> * = nullptr>
  UntypedScalar operator()(Shape &shape,
                           RegionField &in1_rf,
                           RegionField &in2_rf,
                           std::vector<UntypedScalar> &args)
  {
    using OP  = BinaryOp<OP_CODE, CODE>;
    using ARG = legate_type_of<CODE>;

    auto rect = shape.to_rect<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return UntypedScalar(true);

    auto in1 = in1_rf.read_accessor<ARG, DIM>();
    auto in2 = in2_rf.read_accessor<ARG, DIM>();

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = in1.accessor.is_dense_row_major(rect) && in2.accessor.is_dense_row_major(rect);
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
                           RegionField &in2_rf,
                           std::vector<UntypedScalar> &args)
  {
    assert(false);
    return UntypedScalar();
  }
};

struct BinaryOpDispatch {
  template <BinaryOpCode OP_CODE>
  UntypedScalar operator()(Shape &shape,
                           RegionField &in1,
                           RegionField &in2,
                           std::vector<UntypedScalar> &args)
  {
    return double_dispatch(in1.dim(), in1.code(), BinaryOpImpl<OP_CODE>{}, shape, in1, in2, args);
  }
};

}  // namespace omp

/*static*/ UntypedScalar BinaryRedTask::omp_variant(const Task *task,
                                                    const std::vector<PhysicalRegion> &regions,
                                                    Context context,
                                                    Runtime *runtime)
{
  Deserializer ctx(task, regions);

  BinaryOpCode op_code;
  Shape shape;
  RegionField in1;
  RegionField in2;
  std::vector<UntypedScalar> args;

  deserialize(ctx, op_code);
  deserialize(ctx, shape);
  deserialize(ctx, in1);
  deserialize(ctx, in2);
  deserialize(ctx, args);

  return reduce_op_dispatch(op_code, omp::BinaryOpDispatch{}, shape, in1, in2, args);
}

}  // namespace numpy
}  // namespace legate
