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

#include "cunumeric/unary/unary_red.h"
#include "cunumeric/unary/unary_red_template.inl"

namespace cunumeric {

using namespace legate;

struct Split {
  size_t outer{0};
  size_t inner{0};
};

template <int DIM>
class Splitter {
 public:
  Split split(const Rect<DIM>& rect, int must_be_inner)
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

  inline Point<DIM> combine(size_t outer_idx, size_t inner_idx, const Point<DIM>& lo) const
  {
    Point<DIM> point = lo;
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

template <UnaryRedCode OP_CODE, Type CODE, int DIM>
struct UnaryRedImplBody<VariantKind::OMP, OP_CODE, CODE, DIM> {
  using OP    = UnaryRedOp<OP_CODE, CODE>;
  using LG_OP = typename OP::OP;
  using RHS   = legate_type_of<CODE>;

  void operator()(AccessorRD<LG_OP, true, DIM> lhs,
                  AccessorRO<RHS, DIM> rhs,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  int collapsed_dim,
                  size_t volume) const
  {
    Splitter<DIM> splitter;
    auto split = splitter.split(rect, collapsed_dim);

#pragma omp parallel for schedule(static)
    for (size_t o_idx = 0; o_idx < split.outer; ++o_idx)
      for (size_t i_idx = 0; i_idx < split.inner; ++i_idx) {
        auto point = splitter.combine(o_idx, i_idx, rect.lo);
        lhs.reduce(point, OP::convert(point, collapsed_dim, rhs[point]));
      }
  }
};

/*static*/ void UnaryRedTask::omp_variant(TaskContext& context)
{
  unary_red_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
