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

#include "unary_red.h"
#include "unary_red_util.h"
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
  void operator()(Shape &lhs_shape,
                  Shape &rhs_shape,
                  RegionField &lhs_init_rf,
                  RegionField &lhs_red_rf,
                  RegionField &rhs_rf)
  {
    constexpr int LHS_DIM = RHS_DIM - 1;
    using OP              = UnaryRedOp<OP_CODE, CODE>;
    using VAL             = legate_type_of<CODE>;

    Pitches<LHS_DIM - 1> lhs_pitches;
    auto lhs_rect     = lhs_shape.to_rect<LHS_DIM>();
    size_t lhs_volume = lhs_pitches.flatten(lhs_rect);

    if (lhs_volume == 0) return;

    auto lhs_init = lhs_init_rf.write_accessor<VAL, LHS_DIM>();
#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = lhs_init.accessor.is_dense_row_major(lhs_rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    if (dense) {
      auto lhs = lhs_init.ptr(lhs_rect);
      for (size_t idx = 0; idx < lhs_volume; ++idx) lhs[idx] = OP::identity;
    } else {
      for (size_t idx = 0; idx < lhs_volume; ++idx) {
        auto point      = lhs_pitches.unflatten(idx, lhs_rect.lo);
        lhs_init[point] = OP::identity;
      }
    }

    Pitches<RHS_DIM - 1> rhs_pitches;
    auto rhs_rect   = rhs_shape.to_rect<RHS_DIM>();
    auto rhs_volume = rhs_pitches.flatten(rhs_rect);

    auto lhs_red = lhs_red_rf.write_accessor<VAL, RHS_DIM>();
    auto rhs     = rhs_rf.read_accessor<VAL, RHS_DIM>();

    for (size_t idx = 0; idx < rhs_volume; ++idx) {
      auto point = rhs_pitches.unflatten(idx, rhs_rect.lo);
      OP::template fold<true>(lhs_red[point], rhs[point]);
    }
  }

  template <LegateTypeCode CODE,
            int RHS_DIM,
            std::enable_if_t<RHS_DIM <= 1 || !UnaryRedOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(Shape &lhs_shape,
                  Shape &rhs_shape,
                  RegionField &lhs_init,
                  RegionField &lhs_red,
                  RegionField &rhs)
  {
    assert(false);
  }
};

struct UnaryRedDispatch {
  template <UnaryRedCode OP_CODE>
  void operator()(Shape &lhs_shape,
                  Shape &rhs_shape,
                  RegionField &lhs_init,
                  RegionField &lhs_red,
                  RegionField &rhs)
  {
    return double_dispatch(
      rhs.dim(), rhs.code(), UnaryRedImpl<OP_CODE>{}, lhs_shape, rhs_shape, lhs_init, lhs_red, rhs);
  }
};

/*static*/ void UnaryRedTask::cpu_variant(const Task *task,
                                          const std::vector<PhysicalRegion> &regions,
                                          Context context,
                                          Runtime *runtime)
{
  Deserializer ctx(task, regions);

  int32_t collapsed_dim;
  UnaryRedCode op_code;
  Shape lhs_shape;
  Shape rhs_shape;
  // out_init and out_red are aliases of the same region field but with different transformations
  RegionField lhs_init;
  RegionField lhs_red;
  RegionField rhs;

  deserialize(ctx, collapsed_dim);
  deserialize(ctx, op_code);
  deserialize(ctx, lhs_shape);
  deserialize(ctx, rhs_shape);
  deserialize(ctx, lhs_init);
  deserialize(ctx, lhs_red);
  deserialize(ctx, rhs);

  return op_dispatch(op_code, UnaryRedDispatch{}, lhs_shape, rhs_shape, lhs_init, lhs_red, rhs);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { UnaryRedTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
