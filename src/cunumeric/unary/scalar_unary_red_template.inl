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

#include "cunumeric/unary/unary_red_util.h"
#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, UnaryRedCode OP_CODE, LegateTypeCode CODE, int DIM>
struct ScalarUnaryRedImplBody;

template <VariantKind KIND, UnaryRedCode OP_CODE>
struct ScalarUnaryRedImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<UnaryRedOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(ScalarUnaryRedArgs& args) const
  {
    using OP    = UnaryRedOp<OP_CODE, CODE>;
    using LG_OP = typename OP::OP;
    using VAL   = legate_type_of<CODE>;

    auto rect = args.in.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (0 == volume) return;

    auto out = args.out.reduce_accessor<LG_OP, true, 1>();
    auto in  = args.in.read_accessor<VAL, DIM>(rect);

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = in.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    ScalarUnaryRedImplBody<KIND, OP_CODE, CODE, DIM>()(OP{}, out, in, rect, pitches, dense);
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
struct ScalarUnaryRedImpl<KIND, UnaryRedCode::ALL> {
  template <LegateTypeCode CODE, int DIM>
  void operator()(ScalarUnaryRedArgs& args) const
  {
    using OP    = UnaryRedOp<UnaryRedCode::PROD, LegateTypeCode::BOOL_LT>;
    using LG_OP = typename OP::OP;
    using VAL   = legate_type_of<CODE>;

    auto rect = args.in.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (0 == volume) return;

    auto out = args.out.reduce_accessor<LG_OP, true, 1>();
    auto in  = args.in.read_accessor<VAL, DIM>(rect);

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = in.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    ScalarUnaryRedImplBody<KIND, UnaryRedCode::ALL, CODE, DIM>()(out, in, rect, pitches, dense);
  }
};

template <VariantKind KIND>
struct ScalarUnaryRedImpl<KIND, UnaryRedCode::ANY> {
  template <LegateTypeCode CODE, int DIM>
  void operator()(ScalarUnaryRedArgs& args) const
  {
    using OP    = UnaryRedOp<UnaryRedCode::SUM, LegateTypeCode::BOOL_LT>;
    using LG_OP = typename OP::OP;
    using VAL   = legate_type_of<CODE>;

    auto rect = args.in.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (0 == volume) return;

    auto out = args.out.reduce_accessor<LG_OP, true, 1>();
    auto in  = args.in.read_accessor<VAL, DIM>(rect);

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = in.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    ScalarUnaryRedImplBody<KIND, UnaryRedCode::ANY, CODE, DIM>()(out, in, rect, pitches, dense);
  }
};

template <VariantKind KIND>
struct ScalarUnaryRedImpl<KIND, UnaryRedCode::CONTAINS> {
  template <LegateTypeCode CODE, int DIM>
  void operator()(ScalarUnaryRedArgs& args) const
  {
    using OP    = UnaryRedOp<UnaryRedCode::SUM, LegateTypeCode::BOOL_LT>;
    using LG_OP = typename OP::OP;
    using VAL   = legate_type_of<CODE>;

    auto rect = args.in.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (0 == volume) return;

    auto out = args.out.reduce_accessor<LG_OP, true, 1>();
    auto in  = args.in.read_accessor<VAL, DIM>(rect);

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = in.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    ScalarUnaryRedImplBody<KIND, UnaryRedCode::CONTAINS, CODE, DIM>()(
      out, in, args.args[0], rect, pitches, dense);
  }
};

template <VariantKind KIND>
struct ScalarUnaryRedImpl<KIND, UnaryRedCode::COUNT_NONZERO> {
  template <LegateTypeCode CODE, int DIM>
  void operator()(ScalarUnaryRedArgs& args) const
  {
    using OP    = UnaryRedOp<UnaryRedCode::SUM, LegateTypeCode::UINT64_LT>;
    using LG_OP = typename OP::OP;
    using VAL   = legate_type_of<CODE>;

    auto rect = args.in.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (0 == volume) return;

    auto out = args.out.reduce_accessor<LG_OP, true, 1>();
    auto in  = args.in.read_accessor<VAL, DIM>(rect);

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = in.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    ScalarUnaryRedImplBody<KIND, UnaryRedCode::COUNT_NONZERO, CODE, DIM>()(
      out, in, rect, pitches, dense);
  }
};

template <VariantKind KIND>
struct ScalarUnaryRedDispatch {
  template <UnaryRedCode OP_CODE, std::enable_if_t<!is_arg_reduce<OP_CODE>::value>* = nullptr>
  void operator()(ScalarUnaryRedArgs& args) const
  {
    auto dim = std::max(1, args.in.dim());
    double_dispatch(dim, args.in.code(), ScalarUnaryRedImpl<KIND, OP_CODE>{}, args);
  }
  template <UnaryRedCode OP_CODE, std::enable_if_t<is_arg_reduce<OP_CODE>::value>* = nullptr>
  void operator()(ScalarUnaryRedArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void scalar_unary_red_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& scalars = context.scalars();

  std::vector<Store> extra_args;
  for (size_t idx = 1; idx < inputs.size(); ++idx) extra_args.push_back(std::move(inputs[idx]));

  ScalarUnaryRedArgs args{
    context.reductions()[0], inputs[0], scalars[0].value<UnaryRedCode>(), std::move(extra_args)};
  if (args.op_code == UnaryRedCode::COUNT_NONZERO) {
    auto dim = std::max(1, args.in.dim());
    double_dispatch(
      dim, args.in.code(), ScalarUnaryRedImpl<KIND, UnaryRedCode::COUNT_NONZERO>{}, args);
  } else
    op_dispatch(args.op_code, ScalarUnaryRedDispatch<KIND>{}, args);
}

}  // namespace cunumeric
