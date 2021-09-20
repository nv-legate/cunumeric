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

#include "double_binary/double_binary_op_util.h"
#include "pitches.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <VariantKind KIND, DoubleBinaryOpCode OP_CODE, LegateTypeCode CODE, int DIM>
struct DoubleBinaryOpImplBody;

template <VariantKind KIND, DoubleBinaryOpCode OP_CODE>
struct DoubleBinaryOpImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<DoubleBinaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(DoubleBinaryOpArgs& args) const
  {
    using OP  = DoubleBinaryOp<OP_CODE, CODE>;
    using ARG = legate_type_of<CODE>;
    using RES = std::result_of_t<OP(ARG, ARG)>;

    auto rect = args.out.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out = args.out.write_accessor<RES, DIM>(rect);
    auto temp = args.temp.read_write_accessor<RES, DIM>(rect);
    auto in1 = args.in1.read_accessor<ARG, DIM>(rect);
    auto in2 = args.in2.read_accessor<ARG, DIM>(rect);
    auto in3 = args.in2.read_accessor<ARG, DIM>(rect);

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in1.accessor.is_dense_row_major(rect) &&
                 in2.accessor.is_dense_row_major(rect) && in3.accessor.is_dense_row_major(rect) &&
                 temp.accessor.is_dense_row_major(rect); 
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    OP func{args.args};
    DoubleBinaryOpImplBody<KIND, OP_CODE, CODE, DIM>()(func, out,temp, in1, in2,in3, pitches, rect, dense);
  }

  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<!DoubleBinaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(DoubleBinaryOpArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct DoubleBinaryOpDispatch {
  template <DoubleBinaryOpCode OP_CODE>
  void operator()(DoubleBinaryOpArgs& args) const
  {
    auto dim = std::max(args.in1.dim(), args.in2.dim());
    dim = std::max(args.in3.dim(), dim);
    double_dispatch(dim, args.in1.code(), DoubleBinaryOpImpl<KIND, OP_CODE>{}, args);
  }
};

template <VariantKind KIND>
static void double_binary_op_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();
  auto& scalars = context.scalars();

  std::vector<UntypedScalar> extra_args;
  for (size_t idx = 3; idx < inputs.size(); ++idx)
    extra_args.push_back(inputs[idx].scalar<UntypedScalar>());

  DoubleBinaryOpArgs args{
    inputs[0], inputs[1], inputs[2], outputs[0], outputs[1], scalars[0].value<DoubleBinaryOpCode>(), std::move(extra_args)};
  op_dispatch(args.op_code, DoubleBinaryOpDispatch<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
