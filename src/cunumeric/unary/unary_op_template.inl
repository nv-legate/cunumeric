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
#include "cunumeric/unary/unary_op.h"
#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, UnaryOpCode OP_CODE, LegateTypeCode CODE, int DIM>
struct UnaryOpImplBody;

template <VariantKind KIND, typename VAL, int DIM>
struct PointCopyImplBody;

template <VariantKind KIND, UnaryOpCode OP_CODE, LegateTypeCode CODE, int DIM>
struct MultiOutUnaryOpImplBody;

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

#ifndef LEGATE_BOUNDS_CHECKS
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

template <VariantKind KIND, UnaryOpCode OP_CODE>
struct MultiOutUnaryOpImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<MultiOutUnaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(MultiOutUnaryOpArgs& args) const
  {
    using OP   = MultiOutUnaryOp<OP_CODE, CODE>;
    using RHS1 = typename OP::RHS1;
    using RHS2 = typename OP::RHS2;
    using LHS  = std::result_of_t<OP(RHS1, RHS2*)>;

    auto rect = args.out1.shape<DIM>().intersection(args.in.shape<DIM>());

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto lhs  = args.out1.write_accessor<LHS, DIM>(rect);
    auto rhs1 = args.in.read_accessor<RHS1, DIM>(rect);
    auto rhs2 = args.out2.write_accessor<RHS2, DIM>(rect);

#ifndef LEGATE_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = lhs.accessor.is_dense_row_major(rect) && rhs1.accessor.is_dense_row_major(rect) &&
                 rhs2.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    OP func{};
    MultiOutUnaryOpImplBody<KIND, OP_CODE, CODE, DIM>()(
      func, lhs, rhs1, rhs2, pitches, rect, dense);
  }

  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<!MultiOutUnaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(MultiOutUnaryOpArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct UnaryCopyImpl {
  template <LegateTypeCode CODE, int DIM>
  void operator()(UnaryOpArgs& args) const
  {
    using VAL = legate_type_of<CODE>;
    execute_copy<VAL, DIM>(args);
  }

  template <CuNumericTypeCodes CODE, int DIM>
  void operator()(UnaryOpArgs& args) const
  {
    using VAL = cunumeric_type_of<CODE>;
    execute_copy<VAL, DIM>(args);
  }

  template <typename VAL, int DIM>
  void execute_copy(UnaryOpArgs& args) const
  {
    auto rect = args.out.shape<DIM>().intersection(args.in.shape<DIM>());

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out = args.out.write_accessor<VAL, DIM>(rect);
    auto in  = args.in.read_accessor<VAL, DIM>(rect);

#ifndef LEGATE_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    PointCopyImplBody<KIND, VAL, DIM>()(out, in, pitches, rect, dense);
  }
};

template <VariantKind KIND>
struct UnaryOpDispatch {
  template <UnaryOpCode OP_CODE>
  void operator()(UnaryOpArgs& args) const
  {
    auto dim = std::max(args.in.dim(), 1);
    if ((OP_CODE == UnaryOpCode::COPY) &&
        (args.in.code<int32_t>() > LegateTypeCode::MAX_TYPE_NUMBER))
      cunumeric::double_dispatch(
        dim, args.in.code<CuNumericTypeCodes>(), UnaryCopyImpl<KIND>{}, args);
    else
      legate::double_dispatch(dim, args.in.code(), UnaryOpImpl<KIND, OP_CODE>{}, args);
  }
};

template <VariantKind KIND>
static void unary_op_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();
  auto& scalars = context.scalars();

  auto op_code = scalars[0].value<UnaryOpCode>();
  switch (op_code) {
    case UnaryOpCode::FREXP: {
      MultiOutUnaryOpArgs args{inputs[0], outputs[0], outputs[1], op_code};
      auto dim = std::max(args.in.dim(), 1);
      legate::double_dispatch(
        dim, args.in.code(), MultiOutUnaryOpImpl<KIND, UnaryOpCode::FREXP>{}, args);
      break;
    }
    case UnaryOpCode::MODF: {
      MultiOutUnaryOpArgs args{inputs[0], outputs[0], outputs[1], op_code};
      auto dim = std::max(args.in.dim(), 1);
      legate::double_dispatch(
        dim, args.in.code(), MultiOutUnaryOpImpl<KIND, UnaryOpCode::MODF>{}, args);
      break;
    }
    default: {
      std::vector<Store> extra_args;
      for (size_t idx = 1; idx < inputs.size(); ++idx) extra_args.push_back(std::move(inputs[idx]));

      UnaryOpArgs args{inputs[0], outputs[0], op_code, std::move(extra_args)};
      op_dispatch(args.op_code, UnaryOpDispatch<KIND>{}, args);
      break;
    }
  }
}

}  // namespace cunumeric
