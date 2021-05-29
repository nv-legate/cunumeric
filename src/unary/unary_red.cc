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
  using OP     = UnaryRedOp<OP_CODE, CODE>;
  using LG_OP  = typename OP::OP;
  using VAL    = legate_type_of<CODE>;
  using ARGVAL = Argval<VAL>;

  void operator()(AccessorRD<LG_OP, true, DIM> lhs,
                  AccessorRO<VAL, DIM> rhs,
                  const Rect<DIM> &rect,
                  const Pitches<DIM - 1> &pitches,
                  int collapsed_dim,
                  size_t volume) const
  {
    for (size_t idx = 0; idx < volume; ++idx) {
      auto point = pitches.unflatten(idx, rect.lo);
      lhs.reduce(point, ARGVAL(point[collapsed_dim], rhs[point]));
    }
  }

  void operator()(AccessorRW<ARGVAL, DIM> lhs,
                  AccessorRO<VAL, DIM> rhs,
                  const Rect<DIM> &rect,
                  const Pitches<DIM - 1> &pitches,
                  int collapsed_dim,
                  size_t volume) const
  {
    for (size_t idx = 0; idx < volume; ++idx) {
      auto point = pitches.unflatten(idx, rect.lo);
      OP::template fold<true>(lhs[point], ARGVAL(point[collapsed_dim], rhs[point]));
    }
  }
};

void deserialize(Deserializer &ctx, UnaryRedArgs &args)
{
  deserialize(ctx, args.needs_reduction);
  deserialize(ctx, args.collapsed_dim);
  deserialize(ctx, args.op_code);
  deserialize(ctx, args.shape);
  deserialize(ctx, args.lhs);
  deserialize(ctx, args.rhs);
}

/*static*/ void UnaryRedTask::cpu_variant(const Task *task,
                                          const std::vector<PhysicalRegion> &regions,
                                          Context context,
                                          Runtime *runtime)
{
  unary_red_template<VariantKind::CPU>(task, regions, context, runtime);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { UnaryRedTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
