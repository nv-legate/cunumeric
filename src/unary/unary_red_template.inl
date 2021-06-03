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
#include "arg.h"
#include "core.h"
#include "deserializer.h"
#include "dispatch.h"
#include "point_task.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <VariantKind KIND, UnaryRedCode OP_CODE, LegateTypeCode CODE, int DIM>
struct UnaryRedImplBody;

template <VariantKind KIND, UnaryRedCode OP_CODE, LegateTypeCode CODE, int DIM>
struct ArgRedImplBody;

template <VariantKind KIND, UnaryRedCode OP_CODE>
struct UnaryRedImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<(DIM > 1) && UnaryRedOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(UnaryRedArgs &args) const
  {
    using OP  = UnaryRedOp<OP_CODE, CODE>;
    using VAL = legate_type_of<CODE>;

    Pitches<DIM - 1> pitches;
    auto rect   = args.shape.to_rect<DIM>();
    auto volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto rhs = args.rhs.read_accessor<VAL, DIM>(rect);

    if (args.needs_reduction) {
      auto lhs = args.lhs.reduce_accessor<typename OP::OP, KIND != VariantKind::GPU, DIM>(rect);
      UnaryRedImplBody<KIND, OP_CODE, CODE, DIM>()(
        lhs, rhs, rect, pitches, args.collapsed_dim, volume);
    } else {
      auto lhs = args.lhs.read_write_accessor<VAL, DIM>(rect);
      UnaryRedImplBody<KIND, OP_CODE, CODE, DIM>()(
        lhs, rhs, rect, pitches, args.collapsed_dim, volume);
    }
  }

  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<DIM <= 1 || !UnaryRedOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(UnaryRedArgs &args) const
  {
    assert(false);
  }
};

template <VariantKind KIND, UnaryRedCode OP_CODE>
struct ArgRedImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<(DIM > 1) && UnaryRedOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(UnaryRedArgs &args) const
  {
    using OP     = UnaryRedOp<OP_CODE, CODE>;
    using VAL    = legate_type_of<CODE>;
    using ARGVAL = Argval<VAL>;

    Pitches<DIM - 1> pitches;
    auto rect   = args.shape.to_rect<DIM>();
    auto volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto rhs = args.rhs.read_accessor<VAL, DIM>(rect);

    if (args.needs_reduction) {
      auto lhs = args.lhs.reduce_accessor<typename OP::OP, KIND != VariantKind::GPU, DIM>(rect);
      ArgRedImplBody<KIND, OP_CODE, CODE, DIM>()(
        lhs, rhs, rect, pitches, args.collapsed_dim, volume);
    } else {
      auto lhs = args.lhs.read_write_accessor<ARGVAL, DIM>(rect);
      ArgRedImplBody<KIND, OP_CODE, CODE, DIM>()(
        lhs, rhs, rect, pitches, args.collapsed_dim, volume);
    }
  }

  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<DIM <= 1 || !UnaryRedOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(UnaryRedArgs &args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct UnaryRedDispatch {
  template <UnaryRedCode OP_CODE, std::enable_if_t<!is_arg_reduce<OP_CODE>::value> * = nullptr>
  void operator()(UnaryRedArgs &args) const
  {
    return double_dispatch(args.rhs.dim(), args.rhs.code(), UnaryRedImpl<KIND, OP_CODE>{}, args);
  }
  template <UnaryRedCode OP_CODE, std::enable_if_t<is_arg_reduce<OP_CODE>::value> * = nullptr>
  void operator()(UnaryRedArgs &args) const
  {
    return double_dispatch(args.rhs.dim(), args.rhs.code(), ArgRedImpl<KIND, OP_CODE>{}, args);
  }
};

template <VariantKind KIND>
static void unary_red_template(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context context,
                               Runtime *runtime)
{
  Deserializer ctx(task, regions);
  UnaryRedArgs args;
  deserialize(ctx, args);
  op_dispatch(args.op_code, UnaryRedDispatch<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
