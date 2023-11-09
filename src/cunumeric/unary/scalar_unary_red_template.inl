/* Copyright 2021-2023 NVIDIA Corporation
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
#include "cunumeric/cunumeric.h"
#include "cunumeric/unary/scalar_unary_red.h"
#include "cunumeric/unary/unary_red_util.h"
#include "cunumeric/pitches.h"
#include "cunumeric/execution_policy/reduction/scalar_reduction.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, UnaryRedCode OP_CODE, Type::Code CODE, int DIM, bool HAS_WHERE>
struct ScalarUnaryRed {
  ScalarUnaryRed(ScalarUnaryRedArgs& args) { assert(false); }
};

template <VariantKind KIND, UnaryRedCode OP_CODE, Type::Code CODE, int DIM>
struct ScalarUnaryRed<KIND, OP_CODE, CODE, DIM, false> {
  using OP    = UnaryRedOp<OP_CODE, CODE>;
  using LG_OP = typename OP::OP;
  using LHS   = typename OP::VAL;
  using RHS   = legate_type_of<CODE>;
  using OUT   = AccessorRD<LG_OP, true, 1>;
  using IN    = AccessorRO<RHS, DIM>;

  IN in;
  const RHS* inptr;
  OUT out;
  size_t volume;
  Pitches<DIM - 1> pitches;
  Rect<DIM> rect;
  Point<DIM> origin;
  Point<DIM> shape;
  RHS to_find;
  bool dense;

  struct DenseReduction {};
  struct SparseReduction {};

  ScalarUnaryRed(ScalarUnaryRedArgs& args) : dense(false)
  {
    rect   = args.in.shape<DIM>();
    origin = rect.lo;
    in     = args.in.read_accessor<RHS, DIM>(rect);
    volume = pitches.flatten(rect);
    shape  = args.shape;

    out = args.out.reduce_accessor<LG_OP, true, 1>();
    if constexpr (OP_CODE == UnaryRedCode::CONTAINS) { to_find = args.args[0].scalar<RHS>(); }

#ifndef LEGATE_BOUNDS_CHECKS
    // Check to see if this is dense or not
    if (in.accessor.is_dense_row_major(rect)) {
      dense = true;
      inptr = in.ptr(rect);
    }
#endif
  }

  __CUDA_HD__ void operator()(LHS& lhs, size_t idx, LHS identity, DenseReduction) const noexcept
  {
    if constexpr (OP_CODE == UnaryRedCode::CONTAINS) {
      if (inptr[idx] == to_find) { lhs = true; }
    } else if constexpr (OP_CODE == UnaryRedCode::ARGMAX || OP_CODE == UnaryRedCode::ARGMIN ||
                         OP_CODE == UnaryRedCode::NANARGMAX || OP_CODE == UnaryRedCode::NANARGMIN) {
      auto p = pitches.unflatten(idx, origin);
      OP::template fold<true>(lhs, OP::convert(p, shape, identity, inptr[idx]));
    } else {
      OP::template fold<true>(lhs, OP::convert(inptr[idx], identity));
    }
  }

  __CUDA_HD__ void operator()(LHS& lhs, size_t idx, LHS identity, SparseReduction) const noexcept
  {
    if constexpr (OP_CODE == UnaryRedCode::CONTAINS) {
      auto point = pitches.unflatten(idx, origin);
      if (in[point] == to_find) { lhs = true; }
    } else if constexpr (OP_CODE == UnaryRedCode::ARGMAX || OP_CODE == UnaryRedCode::ARGMIN ||
                         OP_CODE == UnaryRedCode::NANARGMAX || OP_CODE == UnaryRedCode::NANARGMIN) {
      auto p = pitches.unflatten(idx, origin);
      OP::template fold<true>(lhs, OP::convert(p, shape, identity, in[p]));
    } else {
      auto p = pitches.unflatten(idx, origin);
      OP::template fold<true>(lhs, OP::convert(in[p], identity));
    }
  }

  void execute() const noexcept
  {
    auto identity = LG_OP::identity;
#ifndef LEGATE_BOUNDS_CHECKS
    // The constexpr if here prevents the DenseReduction from being instantiated for GPU kernels
    // which limits compile times and binary sizes.
    if constexpr (KIND != VariantKind::GPU) {
      // Check to see if this is dense or not
      if (dense) {
        return ScalarReductionPolicy<KIND, LG_OP, DenseReduction>()(volume, out, identity, *this);
      }
    }
#endif
    return ScalarReductionPolicy<KIND, LG_OP, SparseReduction>()(volume, out, identity, *this);
  }
};

template <VariantKind KIND, UnaryRedCode OP_CODE, Type::Code CODE, int DIM>
struct ScalarUnaryRed<KIND, OP_CODE, CODE, DIM, true> {
  using OP    = UnaryRedOp<OP_CODE, CODE>;
  using LG_OP = typename OP::OP;
  using LHS   = typename OP::VAL;
  using RHS   = legate_type_of<CODE>;
  using OUT   = AccessorRD<LG_OP, true, 1>;
  using IN    = AccessorRO<RHS, DIM>;
  using WHERE = AccessorRO<bool, DIM>;

  IN in;
  const RHS* inptr;
  OUT out;
  size_t volume;
  Pitches<DIM - 1> pitches;
  Rect<DIM> rect;
  Point<DIM> origin;
  Point<DIM> shape;
  RHS to_find;
  bool dense;
  WHERE where;
  const bool* whereptr;

  struct DenseReduction {};
  struct SparseReduction {};

  ScalarUnaryRed(ScalarUnaryRedArgs& args) : dense(false)
  {
    rect   = args.in.shape<DIM>();
    origin = rect.lo;
    in     = args.in.read_accessor<RHS, DIM>(rect);
    volume = pitches.flatten(rect);
    shape  = args.shape;

    out = args.out.reduce_accessor<LG_OP, true, 1>();
    if constexpr (OP_CODE == UnaryRedCode::CONTAINS) { to_find = args.args[0].scalar<RHS>(); }

    if ((args.has_where == true) && (OP_CODE != UnaryRedCode::CONTAINS)) {
      where = args.where.read_accessor<bool, DIM>(rect);
#ifndef LEGATE_BOUNDS_CHECKS
      if (in.accessor.is_dense_row_major(rect) && where.accessor.is_dense_row_major(rect)) {
        dense    = true;
        inptr    = in.ptr(rect);
        whereptr = where.ptr(rect);
      }
#endif
    } else {
#ifndef LEGATE_BOUNDS_CHECKS
      // Check to see if this is dense or not
      if (in.accessor.is_dense_row_major(rect)) {
        dense = true;
        inptr = in.ptr(rect);
      }
#endif
    }
  }

  __CUDA_HD__ void operator()(LHS& lhs, size_t idx, LHS identity, DenseReduction) const noexcept
  {
    if constexpr (OP_CODE == UnaryRedCode::CONTAINS) {
      if (inptr[idx] == to_find) { lhs = true; }
    } else if constexpr (OP_CODE == UnaryRedCode::ARGMAX || OP_CODE == UnaryRedCode::ARGMIN ||
                         OP_CODE == UnaryRedCode::NANARGMAX || OP_CODE == UnaryRedCode::NANARGMIN) {
      auto p = pitches.unflatten(idx, origin);
      if (whereptr[idx])
        OP::template fold<true>(lhs, OP::convert(p, shape, identity, inptr[idx]));
    } else {
      if (whereptr[idx]) OP::template fold<true>(lhs, OP::convert(inptr[idx], identity));
    }
  }

  __CUDA_HD__ void operator()(LHS& lhs, size_t idx, LHS identity, SparseReduction) const noexcept
  {
    if constexpr (OP_CODE == UnaryRedCode::CONTAINS) {
      auto point = pitches.unflatten(idx, origin);
      if (in[point] == to_find) { lhs = true; }
    } else if constexpr (OP_CODE == UnaryRedCode::ARGMAX || OP_CODE == UnaryRedCode::ARGMIN ||
                         OP_CODE == UnaryRedCode::NANARGMAX || OP_CODE == UnaryRedCode::NANARGMIN) {
      auto p = pitches.unflatten(idx, origin);
      if (where[p] == true) OP::template fold<true>(lhs, OP::convert(p, shape, identity, in[p]));
    } else {
      auto p = pitches.unflatten(idx, origin);
      if (where[p] == true) OP::template fold<true>(lhs, OP::convert(in[p], identity));
    }
  }

  void execute() const noexcept
  {
    auto identity = LG_OP::identity;
#ifndef LEGATE_BOUNDS_CHECKS
    // The constexpr if here prevents the DenseReduction from being instantiated for GPU kernels
    // which limits compile times and binary sizes.
    if constexpr (KIND != VariantKind::GPU) {
      // Check to see if this is dense or not
      if (dense) {
       ScalarReductionPolicy<KIND, LG_OP, DenseReduction>()(volume, out, identity, *this);
       return;
      }
    }
#endif
   ScalarReductionPolicy<KIND, LG_OP, SparseReduction>()(volume, out, identity, *this);
  }
};

template <VariantKind KIND, UnaryRedCode OP_CODE>
struct ScalarUnaryRedImpl {
  template <Type::Code CODE, int DIM>
  void operator()(ScalarUnaryRedArgs& args) const
  {
    // The operation is always valid for contains
    if constexpr (UnaryRedOp<OP_CODE, CODE>::valid || OP_CODE == UnaryRedCode::CONTAINS) {
      if (args.has_where) {
        ScalarUnaryRed<KIND, OP_CODE, CODE, DIM, true> red(args);
        red.execute();
      } else {
        ScalarUnaryRed<KIND, OP_CODE, CODE, DIM, false> red(args);
        red.execute();
      }
    }
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

  auto op_code     = scalars[0].value<UnaryRedCode>();
  auto shape       = scalars[1].value<DomainPoint>();
  bool has_where   = scalars[2].value<bool>();
  size_t start_idx = has_where ? 2 : 1;
  std::vector<Store> extra_args;
  extra_args.reserve(inputs.size());
  for (size_t idx = start_idx; idx < inputs.size(); ++idx)
     extra_args.emplace_back(std::move(inputs[idx]));
  // If the RHS was a scalar, use (1,) as the shape
  if (shape.dim == 0) {
    shape.dim = 1;
    shape[0]  = 1;
  }

  if (has_where == true) {
    ScalarUnaryRedArgs args{context.reductions()[0],
                            inputs[0],
                            inputs[1],
                            has_where,
                            op_code,
                            shape,
                            std::move(extra_args)};
    op_dispatch(args.op_code, ScalarUnaryRedDispatch<KIND>{}, args);
  } else {
    Array where;
    ScalarUnaryRedArgs args{
      context.reductions()[0], inputs[0], where, has_where, op_code, shape, std::move(extra_args)};
    op_dispatch(args.op_code, ScalarUnaryRedDispatch<KIND>{}, args);
  }
}

}  // namespace cunumeric
