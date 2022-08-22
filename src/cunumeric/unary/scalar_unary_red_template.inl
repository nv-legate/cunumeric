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
#include <core/utilities/typedefs.h>
#include "cunumeric/unary/scalar_unary_red.h"
#include "cunumeric/unary/unary_red_util.h"
#include "cunumeric/pitches.h"
#include "cunumeric/execution_policy/reduction/scalar_reduction.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;


template <VariantKind KIND, UnaryRedCode OP_CODE>
struct ScalarUnaryRedImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<UnaryRedOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(ScalarUnaryRedArgs& args) const
  {
    using RHS   = legate_type_of<CODE>;
    auto rect = args.in.shape<DIM>();
    auto in  = args.in.read_accessor<RHS, DIM>(rect);

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    if (in.accessor.is_dense_row_major(rect)){
      return execute_kernel<true, CODE, DIM>(in, args);
    }
#endif
    return execute_kernel<false, CODE, DIM>(in, args);
  }

  template <bool dense, LegateTypeCode CODE, int DIM, class AccessorRO>
  void execute_kernel(AccessorRO& in, ScalarUnaryRedArgs& args) const {
    using OP    = UnaryRedOp<OP_CODE, CODE>;
    using LG_OP = typename OP::OP;
    using LHS   = typename OP::VAL;
    using RHS   = legate_type_of<CODE>;
    using scalar_reduction_impl::ScalarReductionPolicy;

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
      // The reduction only checks if any element is equal to the search element.
      if constexpr (dense && KIND != VariantKind::GPU) {
        // On CPU, you can directly access through a pointer.
        auto inptr = in.ptr(rect);
        ScalarReductionPolicy<KIND, LG_OP>()(volume, out, /*identity=*/false, [=](bool& lhs, size_t idx) {
          if (inptr[idx] == to_find) { lhs = true; }
        });
      } else {
        // On GPU or if not dense, must go through accessor.
        ScalarReductionPolicy<KIND, LG_OP>()(volume, out, /*identity=*/false, [=] __host__ __device__  (bool& lhs, size_t idx) {
          auto point = pitches.unflatten(idx, origin);
          if (in[point] == to_find) { lhs = true; }
        });
      }
    } else if constexpr (OP_CODE == UnaryRedCode::ARGMAX || OP_CODE == UnaryRedCode::ARGMIN) {
      // The reduction performs a min/max, but records the index of the maximum, not the value.
      if constexpr (dense && KIND != VariantKind::GPU) {
        // On CPU, you can directly access through a pointer.
        auto inptr = in.ptr(rect);
        ScalarReductionPolicy<KIND, LG_OP>()(volume, out, identity, [=](LHS& lhs, size_t idx) {
          auto p = pitches.unflatten(idx, origin);
          OP::template fold<true>(lhs, OP::convert(p, shape, inptr[idx]));
        });
      } else {
        ScalarReductionPolicy<KIND, LG_OP>()(volume, out, identity, [=] __host__ __device__  (LHS& lhs, size_t idx) {
          auto p = pitches.unflatten(idx, origin);
          OP::template fold<true>(lhs, OP::convert(p, shape, in[p]));
        });
      }
    } else {  // All other op types
      if constexpr (dense && KIND != VariantKind::GPU) {
        // On CPU, you can directly access through a pointer.
        auto inptr = in.ptr(rect);
        ScalarReductionPolicy<KIND, LG_OP>()(volume, out, identity, [=](LHS& lhs, size_t idx) {
          OP::template fold<true>(lhs, OP::convert(inptr[idx]));
        });
      } else {
        ScalarReductionPolicy<KIND, LG_OP>()(volume, out, identity, [=] __host__ __device__  (LHS& lhs, size_t idx) {
          auto p = pitches.unflatten(idx, origin);
          OP::template fold<true>(lhs, OP::convert(in[p]));
        });
      }
    }
  }

  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<!UnaryRedOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(ScalarUnaryRedArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct ScalarUnaryRedDispatch {
  template <UnaryRedCode OP_CODE>
  void operator()(ScalarUnaryRedArgs& args) const
  {
    auto dim = std::max(1, args.in.dim());
    double_dispatch(dim, args.in.code(), ScalarUnaryRedImpl<KIND, OP_CODE>{}, args);
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
  op_dispatch(args.op_code, ScalarUnaryRedDispatch<KIND>{}, args);
}

}  // namespace cunumeric
