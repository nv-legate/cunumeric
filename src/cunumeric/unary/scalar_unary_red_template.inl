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
#include <core/utilities/dispatch.h>
#include <core/utilities/typedefs.h>
#include "cunumeric/cunumeric.h"
#include "cunumeric/unary/scalar_unary_red.h"
#include "cunumeric/unary/unary_red_util.h"
#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND>
struct ScalarUnaryRedImplBody;

template <VariantKind KIND,
          UnaryRedCode OP_CODE = UnaryRedCode::UNSPECIFIED,
          LegateTypeCode CODE  = LegateTypeCode::UNSPECIFIED,
          int DIM              = DIM_UNSPECIFIED>
struct ScalarUnaryRedOp {
  using OP  = UnaryRedOp<OP_CODE, CODE>;
  using RHS = legate_type_of<CODE>;

  struct SetOpCodeNttp {
    template <UnaryRedCode _OP_CODE>
    void operator()(ScalarUnaryRedArgs& args)
    {
      ScalarUnaryRedOp<KIND, _OP_CODE, CODE, DIM>()(args);
    }
  };

  struct SetDimNttp {
    template <int _DIM>
    void operator()(ScalarUnaryRedArgs& args)
    {
      ScalarUnaryRedOp<KIND, OP_CODE, CODE, _DIM>()(args);
    }
  };

  struct SetCodeNttp {
    template <LegateTypeCode _CODE>
    void operator()(ScalarUnaryRedArgs& args)
    {
      ScalarUnaryRedOp<KIND, OP_CODE, _CODE, DIM>()(args);
    }
  };

  void operator()(ScalarUnaryRedArgs& args) const
  {
    if constexpr (OP_CODE == UnaryRedCode::UNSPECIFIED) {
      runtime_parameter_to_nttp<SetOpCodeNttp>(args.op_code, args);
    } else if constexpr (CODE == LegateTypeCode::UNSPECIFIED) {
      LegateTypeCode code = args.in.code();
      legate::runtime_parameter_to_nttp<SetCodeNttp>(code, args);
    } else {
      // Only generate a kernel if this is a valid operation.
      if constexpr (UnaryRedOp<OP_CODE, CODE>::valid) {
        if constexpr (DIM == DIM_UNSPECIFIED) {
          // convert the dim into a non-type template parameter
          int dim = std::max(1, args.in.dim());
          legate::runtime_parameter_to_nttp<SetDimNttp>(dim, args);
        } else {
          // All runtime parameters have been converted to non-type template parameters
          // and this is a valid combination of types. Execute it now.
          execute_kernel(args);
        }
      }
    }
  }

  void execute_kernel(ScalarUnaryRedArgs& args) const
  {
    auto rect = args.in.shape<DIM>();
    auto in   = args.in.read_accessor<RHS, DIM>(rect);
#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = in.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    if (dense) {
      configure_accessor<true>(in, args);
    } else {
      configure_accessor<false>(in, args);
    }
  }

  template <bool dense, class AccessorRO>
  void configure_accessor(AccessorRO& in, ScalarUnaryRedArgs& args) const
  {
    using LG_OP = typename OP::OP;
    using LHS   = typename OP::VAL;

    auto rect        = args.in.shape<DIM>();
    auto origin      = rect.lo;
    Point<DIM> shape = args.shape;

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    auto identity = LG_OP::identity;

    if (0 == volume) return;

    auto out = args.out.reduce_accessor<LG_OP, true, 1>();

    if constexpr (OP_CODE == UnaryRedCode::CONTAINS) {
      auto to_find = args.args[0].scalar<RHS>();
      if constexpr (dense && KIND != VariantKind::GPU) {
        // On CPU, you can directly access through a pointer.
        auto inptr = in.ptr(rect);
        ScalarUnaryRedImplBody<KIND>()(out, volume, /*identity=*/false, [=](bool& lhs, size_t idx) {
          if (inptr[idx] == to_find) { lhs = true; }
        });
      } else {
        ScalarUnaryRedImplBody<KIND>()(out, volume, /*identity=*/false, [=](bool& lhs, size_t idx) {
          auto point = pitches.unflatten(idx, origin);
          if (in[point] == to_find) { lhs = true; }
        });
      }
    } else if constexpr (OP_CODE == UnaryRedCode::ARGMAX || OP_CODE == UnaryRedCode::ARGMIN) {
      if constexpr (dense && KIND != VariantKind::GPU) {
        // On CPU, you can directly access through a pointer.
        auto inptr = in.ptr(rect);
        ScalarUnaryRedImplBody<KIND>()(out, volume, identity, [=](LHS& lhs, size_t idx) {
          auto p = pitches.unflatten(idx, origin);
          OP::template fold<true>(lhs, OP::convert(p, shape, inptr[idx]));
        });
      } else {
        ScalarUnaryRedImplBody<KIND>()(out, volume, identity, [=](LHS& lhs, size_t idx) {
          auto p = pitches.unflatten(idx, origin);
          OP::template fold<true>(lhs, OP::convert(p, shape, in[p]));
        });
      }
    } else {  // All other op types
      if constexpr (dense && KIND != VariantKind::GPU) {
        // On CPU, you can directly access through a pointer.
        auto inptr = in.ptr(rect);
        ScalarUnaryRedImplBody<KIND>()(out, volume, identity, [=](LHS& lhs, size_t idx) {
          OP::template fold<true>(lhs, OP::convert(inptr[idx]));
        });
      } else {
        ScalarUnaryRedImplBody<KIND>()(out, volume, identity, [=](LHS& lhs, size_t idx) {
          auto p = pitches.unflatten(idx, origin);
          OP::template fold<true>(lhs, OP::convert(in[p]));
        });
      }
    }
  }
};

template <VariantKind KIND>
static void scalar_unary_red_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& scalars = context.scalars();

  std::vector<Store> extra_args;
  for (size_t idx = 1; idx < inputs.size(); ++idx) extra_args.push_back(std::move(inputs[idx]));

  auto op_code = scalars[0].value<UnaryRedCode>();
  auto shape   = scalars[1].value<DomainPoint>();
  // If the RHS was a scalar, use (1,) as the shape
  if (shape.dim == 0) {
    shape.dim = 1;
    shape[0]  = 1;
  }
  ScalarUnaryRedArgs args{
    context.reductions()[0], inputs[0], op_code, shape, std::move(extra_args)};

  ScalarUnaryRedOp<KIND>()(args);
}

}  // namespace cunumeric
