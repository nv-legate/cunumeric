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
#include "unary/unary_red_template.inl"

namespace legate {
namespace numpy {

using namespace Legion;

template <UnaryRedCode OP_CODE, LegateTypeCode CODE, int DIM>
struct UnaryRedImplBody<VariantKind::CPU, OP_CODE, CODE, DIM> {
  using OP    = UnaryRedOp<OP_CODE, CODE>;
  using LG_OP = typename OP::OP;
  using VAL   = legate_type_of<CODE>;

  void operator()(AccessorRD<LG_OP, true, DIM> lhs,
                  AccessorRO<VAL, DIM> rhs,
                  const Rect<DIM> &rect,
                  const Pitches<DIM - 1> &pitches,
                  int collapsed_dim,
                  size_t volume) const
  {
    for (size_t idx = 0; idx < volume; ++idx) {
      auto point = pitches.unflatten(idx, rect.lo);
      lhs.reduce(point, rhs[point]);
    }
  }

  void operator()(AccessorRW<VAL, DIM> lhs,
                  AccessorRO<VAL, DIM> rhs,
                  const Rect<DIM> &rect,
                  const Pitches<DIM - 1> &pitches,
                  int collapsed_dim,
                  size_t volume) const
  {
    for (size_t idx = 0; idx < volume; ++idx) {
      auto point = pitches.unflatten(idx, rect.lo);
      OP::template fold<true>(lhs[point], rhs[point]);
    }
  }
};

template <UnaryRedCode OP_CODE, LegateTypeCode CODE, int DIM>
struct ArgRedImplBody<VariantKind::CPU, OP_CODE, CODE, DIM> {
  using OP    = UnaryRedOp<OP_CODE, CODE>;
  using LG_OP = typename OP::OP;
  using RHS   = legate_type_of<CODE>;
  using LHS   = Argval<RHS>;

  void operator()(AccessorRD<LG_OP, true, DIM> lhs,
                  AccessorRO<RHS, DIM> rhs,
                  const Rect<DIM> &rect,
                  const Pitches<DIM - 1> &pitches,
                  int collapsed_dim,
                  size_t volume) const
  {
    for (size_t idx = 0; idx < volume; ++idx) {
      auto point = pitches.unflatten(idx, rect.lo);
      lhs.reduce(point, LHS(point[collapsed_dim], rhs[point]));
    }
  }

  void operator()(AccessorRW<LHS, DIM> lhs,
                  AccessorRO<RHS, DIM> rhs,
                  const Rect<DIM> &rect,
                  const Pitches<DIM - 1> &pitches,
                  int collapsed_dim,
                  size_t volume) const
  {
    for (size_t idx = 0; idx < volume; ++idx) {
      auto point = pitches.unflatten(idx, rect.lo);
      OP::template fold<true>(lhs[point], LHS(point[collapsed_dim], rhs[point]));
    }
  }
};

/*static*/ void UnaryRedTask::cpu_variant(TaskContext &context)
{
  unary_red_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { UnaryRedTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
