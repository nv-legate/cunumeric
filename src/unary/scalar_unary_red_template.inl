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

#include "unary/unary_red_util.h"
#include "core.h"
#include "deserializer.h"
#include "dispatch.h"
#include "point_task.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <VariantKind KIND, UnaryRedCode OP_CODE, LegateTypeCode CODE, int DIM>
struct ScalarUnaryRedImplBody;

template <VariantKind KIND, UnaryRedCode OP_CODE>
struct ScalarUnaryRedImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<UnaryRedOp<OP_CODE, CODE>::valid> * = nullptr>
  UntypedScalar operator()(ScalarUnaryRedArgs &args) const
  {
    using OP  = UnaryRedOp<OP_CODE, CODE>;
    using VAL = legate_type_of<CODE>;

    auto rect = args.shape.to_rect<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    VAL result = OP::identity;

    if (volume == 0) return UntypedScalar(result);

    auto in = args.in.read_accessor<VAL, DIM>();

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = in.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    ScalarUnaryRedImplBody<KIND, OP_CODE, CODE, DIM>()(OP{}, result, in, rect, pitches, dense);

    return UntypedScalar(result);
  }

  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<!UnaryRedOp<OP_CODE, CODE>::valid> * = nullptr>
  UntypedScalar operator()(ScalarUnaryRedArgs &args) const
  {
    assert(false);
    return UntypedScalar();
  }
};

template <VariantKind KIND>
struct ScalarUnaryRedImpl<KIND, UnaryRedCode::CONTAINS> {
  template <LegateTypeCode CODE, int DIM>
  UntypedScalar operator()(ScalarUnaryRedArgs &args) const
  {
    using VAL = legate_type_of<CODE>;

    auto rect = args.shape.to_rect<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    bool result = false;

    if (volume == 0) return UntypedScalar(result);

    auto in = args.in.read_accessor<VAL, DIM>();

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = in.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    ScalarUnaryRedImplBody<KIND, UnaryRedCode::CONTAINS, CODE, DIM>()(
      result, in, args.args[0], rect, pitches, dense);

    return UntypedScalar(result);
  }
};

template <VariantKind KIND>
struct ScalarUnaryRedDispatch {
  template <UnaryRedCode OP_CODE, std::enable_if_t<!is_arg_reduce<OP_CODE>::value> * = nullptr>
  UntypedScalar operator()(ScalarUnaryRedArgs &args) const
  {
    return double_dispatch(
      args.in.dim(), args.in.code(), ScalarUnaryRedImpl<KIND, OP_CODE>{}, args);
  }
  template <UnaryRedCode OP_CODE, std::enable_if_t<is_arg_reduce<OP_CODE>::value> * = nullptr>
  UntypedScalar operator()(ScalarUnaryRedArgs &args) const
  {
    assert(false);
    return UntypedScalar();
  }
};

template <VariantKind KIND>
static UntypedScalar scalar_unary_red_template(const Task *task,
                                               const std::vector<PhysicalRegion> &regions,
                                               Context context,
                                               Runtime *runtime)
{
  Deserializer ctx(task, regions);
  ScalarUnaryRedArgs args;
  deserialize(ctx, args);
  return op_dispatch(args.op_code, ScalarUnaryRedDispatch<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
