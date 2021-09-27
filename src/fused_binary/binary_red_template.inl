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

#include "binary/binary_op_util.h"
#include "pitches.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <VariantKind KIND, BinaryOpCode OP_CODE, LegateTypeCode CODE, int DIM>
struct BinaryRedImplBody;

template <VariantKind KIND, BinaryOpCode OP_CODE>
struct BinaryRedImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<BinaryOp<OP_CODE, CODE>::valid>* = nullptr>
  UntypedScalar operator()(BinaryRedArgs& args) const
  {
    using OP  = BinaryOp<OP_CODE, CODE>;
    using ARG = legate_type_of<CODE>;

    auto rect = args.in1.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return UntypedScalar(true);

    auto in1 = args.in1.read_accessor<ARG, DIM>(rect);
    auto in2 = args.in2.read_accessor<ARG, DIM>(rect);

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = in1.accessor.is_dense_row_major(rect) && in2.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    OP func(args.args);
    return BinaryRedImplBody<KIND, OP_CODE, CODE, DIM>()(func, in1, in2, pitches, rect, dense);
  }

  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<!BinaryOp<OP_CODE, CODE>::valid>* = nullptr>
  UntypedScalar operator()(BinaryRedArgs& args) const
  {
    assert(false);
    return UntypedScalar();
  }
};

template <VariantKind KIND>
struct BinaryRedDispatch {
  template <BinaryOpCode OP_CODE>
  UntypedScalar operator()(BinaryRedArgs& args) const
  {
    auto dim = std::max(args.in1.dim(), args.in2.dim());
    return double_dispatch(dim, args.in1.code(), BinaryRedImpl<KIND, OP_CODE>{}, args);
  }
};

template <VariantKind KIND>
static UntypedScalar binary_red_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& scalars = context.scalars();

  std::vector<UntypedScalar> extra_args;
  for (size_t idx = 2; idx < inputs.size(); ++idx)
    extra_args.push_back(inputs[idx].scalar<UntypedScalar>());

  BinaryRedArgs args{inputs[0], inputs[1], scalars[0].value<BinaryOpCode>(), std::move(extra_args)};
  return reduce_op_dispatch(args.op_code, BinaryRedDispatch<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
