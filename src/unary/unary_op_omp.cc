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

#include "unary/unary_op.h"
#include "core.h"
#include "deserializer.h"
#include "dispatch.h"
#include "point_task.h"

namespace legate {
namespace numpy {

using namespace Legion;

namespace omp {

template <UnaryOpCode OP_CODE>
struct UnaryOpImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<UnaryOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(Shape &shape, RegionField &out_rf, RegionField &in_rf)
  {
    using OP  = UnaryOp<OP_CODE, CODE>;
    using ARG = legate_type_of<CODE>;
    using RES = std::result_of_t<OP(ARG)>;

    auto rect = shape.to_rect<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out = out_rf.write_accessor<RES, DIM>();
    auto in  = in_rf.read_accessor<ARG, DIM>();

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    OP func{};
    if (dense) {
      auto outptr = out.ptr(rect);
      auto inptr  = in.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) outptr[idx] = func(inptr[idx]);
    } else {
      OMPLoop<DIM>::unary_loop(func, out, in, rect);
    }
  }

  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<!UnaryOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(Shape &shape, RegionField &out_rf, RegionField &in_rf)
  {
    assert(false);
  }
};

struct UnaryOpDispatch {
  template <UnaryOpCode OP_CODE>
  void operator()(Shape &shape, RegionField &out, RegionField &in)
  {
    double_dispatch(in.dim(), in.code(), UnaryOpImpl<OP_CODE>{}, shape, out, in);
  }
};

}  // namespace omp

/*static*/ void UnaryOpTask::omp_variant(const Task *task,
                                         const std::vector<PhysicalRegion> &regions,
                                         Context context,
                                         Runtime *runtime)
{
  Deserializer ctx(task, regions);

  UnaryOpCode op_code;
  Shape shape;
  RegionField out;
  RegionField in;

  deserialize(ctx, op_code);
  deserialize(ctx, shape);
  deserialize(ctx, out);
  deserialize(ctx, in);

  op_dispatch(op_code, omp::UnaryOpDispatch{}, shape, out, in);
}

}  // namespace numpy
}  // namespace legate
