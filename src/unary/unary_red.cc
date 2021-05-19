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

#include "unary/unary_red.h"
#include "unary/unary_red_util.h"
#include "core.h"
#include "deserializer.h"
#include "dispatch.h"
#include "point_task.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <UnaryRedCode OP_CODE>
struct UnaryRedImpl {
  template <LegateTypeCode CODE,
            int RHS_DIM,
            std::enable_if_t<(RHS_DIM > 1) && UnaryRedOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(Shape &rhs_shape, RegionField &lhs_rf, RegionField &rhs_rf, bool needs_reduction)
  {
    constexpr int LHS_DIM = RHS_DIM - 1;
    using OP              = UnaryRedOp<OP_CODE, CODE>;
    using VAL             = legate_type_of<CODE>;

    Pitches<RHS_DIM - 1> rhs_pitches;
    auto rhs_rect   = rhs_shape.to_rect<RHS_DIM>();
    auto rhs_volume = rhs_pitches.flatten(rhs_rect);

    if (rhs_volume == 0) return;

    auto rhs = rhs_rf.read_accessor<VAL, RHS_DIM>();

    if (needs_reduction) {
      auto lhs = lhs_rf.reduce_accessor<typename OP::OP, true, RHS_DIM>();
      for (size_t idx = 0; idx < rhs_volume; ++idx) {
        auto point = rhs_pitches.unflatten(idx, rhs_rect.lo);
        lhs.reduce(point, rhs[point]);
      }
    } else {
      auto lhs = lhs_rf.write_accessor<VAL, RHS_DIM>();
      for (size_t idx = 0; idx < rhs_volume; ++idx) {
        auto point = rhs_pitches.unflatten(idx, rhs_rect.lo);
        OP::template fold<true>(lhs[point], rhs[point]);
      }
    }
  }

  template <LegateTypeCode CODE,
            int RHS_DIM,
            std::enable_if_t<RHS_DIM <= 1 || !UnaryRedOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(Shape &rhs_shape, RegionField &lhs, RegionField &rhs, bool needs_reduction)
  {
    assert(false);
  }
};

struct UnaryRedDispatch {
  template <UnaryRedCode OP_CODE>
  void operator()(Shape &rhs_shape, RegionField &lhs, RegionField &rhs, bool needs_reduction)
  {
    return double_dispatch(
      rhs.dim(), rhs.code(), UnaryRedImpl<OP_CODE>{}, rhs_shape, lhs, rhs, needs_reduction);
  }
};

/*static*/ void UnaryRedTask::cpu_variant(const Task *task,
                                          const std::vector<PhysicalRegion> &regions,
                                          Context context,
                                          Runtime *runtime)
{
  Deserializer ctx(task, regions);

  bool needs_reduction;
  int32_t collapsed_dim;
  UnaryRedCode op_code;
  Shape rhs_shape;
  RegionField lhs;
  RegionField rhs;

  deserialize(ctx, needs_reduction);
  deserialize(ctx, collapsed_dim);
  deserialize(ctx, op_code);
  deserialize(ctx, rhs_shape);
  deserialize(ctx, lhs);
  deserialize(ctx, rhs);

  return op_dispatch(op_code, UnaryRedDispatch{}, rhs_shape, lhs, rhs, needs_reduction);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { UnaryRedTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
