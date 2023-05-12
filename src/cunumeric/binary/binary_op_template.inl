/* Copyright 2021-2022 NVIDIA Corporation
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

#pragma once

// Useful for IDEs
#include "cunumeric/binary/binary_op.h"
#include "cunumeric/binary/binary_op_util.h"
#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, BinaryOpCode OP_CODE, Type::Code CODE, int DIM>
struct BinaryOpImplBody;

template <VariantKind KIND, BinaryOpCode OP_CODE>
struct BinaryOpImpl {
  template <Type::Code CODE, int DIM, std::enable_if_t<BinaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(BinaryOpArgs& args) const
  {
    using OP   = BinaryOp<OP_CODE, CODE>;
    using RHS1 = legate_type_of<CODE>;
    using RHS2 = rhs2_of_binary_op<OP_CODE, CODE>;
    using LHS  = std::result_of_t<OP(RHS1, RHS2)>;

    auto rect = args.out.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out = args.out.write_accessor<LHS, DIM>(rect);
    auto in1 = args.in1.read_accessor<RHS1, DIM>(rect);
    auto in2 = args.in2.read_accessor<RHS2, DIM>(rect);

#ifndef LEGATE_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in1.accessor.is_dense_row_major(rect) &&
                 in2.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    OP func{args.args};
    BinaryOpImplBody<KIND, OP_CODE, CODE, DIM>()(func, out, in1, in2, pitches, rect, dense);
  }

  template <Type::Code CODE, int DIM, std::enable_if_t<!BinaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(BinaryOpArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct BinaryOpDispatch {
  template <BinaryOpCode OP_CODE>
  void operator()(BinaryOpArgs& args) const
  {
    auto dim = std::max(1, args.out.dim());
    double_dispatch(dim, args.in1.code(), BinaryOpImpl<KIND, OP_CODE>{}, args);
  }
};

template <VariantKind KIND>
static void binary_op_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();
  auto& scalars = context.scalars();

  std::vector<Store> extra_args;
  for (size_t idx = 2; idx < inputs.size(); ++idx) extra_args.push_back(std::move(inputs[idx]));

  BinaryOpArgs args{
    inputs[0], inputs[1], outputs[0], scalars[0].value<BinaryOpCode>(), std::move(extra_args)};
  op_dispatch(args.op_code, BinaryOpDispatch<KIND>{}, args);
}

}  // namespace cunumeric
