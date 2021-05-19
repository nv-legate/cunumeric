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

namespace omp {

struct Split {
  size_t outer{0};
  size_t inner{0};
};

template <int DIM>
class Splitter {
 public:
  Split split(const Legion::Rect<DIM> &rect, int must_be_inner)
  {
    for (int dim = 0; dim < DIM; ++dim)
      if (dim != must_be_inner) {
        outer_dim_ = dim;
        break;
      }

    size_t outer = 1;
    size_t inner = 1;
    size_t pitch = 1;
    for (int dim = DIM - 1; dim >= 0; --dim) {
      auto diff = rect.hi[dim] - rect.lo[dim] + 1;
      if (dim == outer_dim_)
        outer *= diff;
      else {
        inner *= diff;
        pitches_[dim] = pitch;
        pitch *= diff;
      }
    }
    return Split{outer, inner};
  }

  inline Legion::Point<DIM> combine(size_t outer_idx,
                                    size_t inner_idx,
                                    const Legion::Point<DIM> &lo) const
  {
    Legion::Point<DIM> point = lo;
    for (int dim = 0; dim < DIM; ++dim) {
      if (dim == outer_dim_)
        point[dim] += outer_idx;
      else {
        point[dim] += inner_idx / pitches_[dim];
        inner_idx = inner_idx % pitches_[dim];
      }
    }
    return point;
  }

 private:
  int32_t outer_dim_;
  size_t pitches_[DIM];
};

template <UnaryRedCode OP_CODE>
struct UnaryRedImpl {
  template <LegateTypeCode CODE,
            int RHS_DIM,
            std::enable_if_t<(RHS_DIM > 1) && UnaryRedOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(int collapsed_dim,
                  Shape &rhs_shape,
                  RegionField &lhs_rf,
                  RegionField &rhs_rf,
                  bool needs_reduction)
  {
    constexpr int LHS_DIM = RHS_DIM - 1;
    using OP              = UnaryRedOp<OP_CODE, CODE>;
    using VAL             = legate_type_of<CODE>;

    Splitter<RHS_DIM> rhs_splitter;
    auto rhs_rect = rhs_shape.to_rect<RHS_DIM>();
    if (rhs_rect.volume() == 0) return;
    auto rhs_split = rhs_splitter.split(rhs_rect, collapsed_dim);

    auto rhs = rhs_rf.read_accessor<VAL, RHS_DIM>();

    if (needs_reduction) {
      auto lhs = lhs_rf.reduce_accessor<typename OP::OP, true, RHS_DIM>();
#pragma omp parallel for schedule(static)
      for (size_t o_idx = 0; o_idx < rhs_split.outer; ++o_idx)
        for (size_t i_idx = 0; i_idx < rhs_split.inner; ++i_idx) {
          auto point = rhs_splitter.combine(o_idx, i_idx, rhs_rect.lo);
          lhs.reduce(point, rhs[point]);
        }
    } else {
      auto lhs = lhs_rf.read_write_accessor<VAL, RHS_DIM>();
#pragma omp parallel for schedule(static)
      for (size_t o_idx = 0; o_idx < rhs_split.outer; ++o_idx)
        for (size_t i_idx = 0; i_idx < rhs_split.inner; ++i_idx) {
          auto point = rhs_splitter.combine(o_idx, i_idx, rhs_rect.lo);
          OP::template fold<true>(lhs[point], rhs[point]);
        }
    }
  }

  template <LegateTypeCode CODE,
            int RHS_DIM,
            std::enable_if_t<RHS_DIM <= 1 || !UnaryRedOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(int collapsed_dim,
                  Shape &rhs_shape,
                  RegionField &lhs_rf,
                  RegionField &rhs_rf,
                  bool needs_reduction)
  {
    assert(false);
  }
};

struct UnaryRedDispatch {
  template <UnaryRedCode OP_CODE>
  void operator()(
    int collapsed_dim, Shape &rhs_shape, RegionField &lhs, RegionField &rhs, bool needs_reduction)
  {
    return double_dispatch(rhs.dim(),
                           rhs.code(),
                           UnaryRedImpl<OP_CODE>{},
                           collapsed_dim,
                           rhs_shape,
                           lhs,
                           rhs,
                           needs_reduction);
  }
};

}  // namespace omp

/*static*/ void UnaryRedTask::omp_variant(const Task *task,
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

  return op_dispatch(
    op_code, omp::UnaryRedDispatch{}, collapsed_dim, rhs_shape, lhs, rhs, needs_reduction);
}

}  // namespace numpy
}  // namespace legate
