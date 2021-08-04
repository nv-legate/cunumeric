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

#include "point_task.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <VariantKind KIND, UnaryOpCode OP_CODE, LegateTypeCode CODE, int DIM>
struct UnaryOpImplBody;

template <VariantKind KIND, UnaryOpCode OP_CODE>
struct UnaryOpImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<UnaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(UnaryOpArgs& args) const
  {
    using OP  = UnaryOp<OP_CODE, CODE>;
    using ARG = typename OP::T;
    using RES = std::result_of_t<OP(ARG)>;

    auto rect = args.out.shape<DIM>().intersection(args.in.shape<DIM>());

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out = args.out.write_accessor<RES, DIM>(rect);
    auto in  = args.in.read_accessor<ARG, DIM>(rect);

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    OP func{args.args};
    UnaryOpImplBody<KIND, OP_CODE, CODE, DIM>()(func, out, in, pitches, rect, dense);
  }

  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<!UnaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(UnaryOpArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct UnaryOpDispatch {
  template <UnaryOpCode OP_CODE>
  void operator()(UnaryOpArgs& args) const
  {
    double_dispatch(args.in.dim(), args.in.code(), UnaryOpImpl<KIND, OP_CODE>{}, args);
  }
};

template <VariantKind KIND>
static void unary_op_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();
  auto& scalars = context.scalars();

  std::vector<UntypedScalar> extra_args;
  for (size_t idx = 1; idx < inputs.size(); ++idx)
    extra_args.push_back(inputs[idx].scalar<UntypedScalar>());

  UnaryOpArgs args{inputs[0], outputs[0], scalars[0].value<UnaryOpCode>(), std::move(extra_args)};
  op_dispatch(args.op_code, UnaryOpDispatch<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
