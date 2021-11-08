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

#include "cunumeric/unary/scalar_unary_red.h"
#include "cunumeric/unary/scalar_unary_red_template.inl"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <UnaryRedCode OP_CODE, LegateTypeCode CODE, int DIM>
struct ScalarUnaryRedImplBody<VariantKind::CPU, OP_CODE, CODE, DIM> {
  using OP    = UnaryRedOp<OP_CODE, CODE>;
  using LG_OP = typename OP::OP;
  using VAL   = legate_type_of<CODE>;

  void operator()(OP func,
                  AccessorRD<LG_OP, true, 1> out,
                  AccessorRO<VAL, DIM> in,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  bool dense) const
  {
    auto result         = LG_OP::identity;
    const size_t volume = rect.volume();
    if (dense) {
      auto inptr = in.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) OP::template fold<true>(result, inptr[idx]);
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        OP::template fold<true>(result, in[p]);
      }
    }
    out.reduce(0, result);
  }
};

template <LegateTypeCode CODE, int DIM>
struct ScalarUnaryRedImplBody<VariantKind::CPU, UnaryRedCode::CONTAINS, CODE, DIM> {
  using OP    = UnaryRedOp<UnaryRedCode::SUM, LegateTypeCode::BOOL_LT>;
  using LG_OP = typename OP::OP;
  using VAL   = legate_type_of<CODE>;

  void operator()(AccessorRD<LG_OP, true, 1> out,
                  AccessorRO<VAL, DIM> in,
                  const Store& to_find_scalar,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  bool dense) const
  {
    auto result         = LG_OP::identity;
    const auto to_find  = to_find_scalar.scalar<VAL>();
    const size_t volume = rect.volume();
    if (dense) {
      auto inptr = in.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx)
        if (inptr[idx] == to_find) {
          result = true;
          break;
        }
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto point = pitches.unflatten(idx, rect.lo);
        if (in[point] == to_find) {
          result = true;
          break;
        }
      }
    }
    out.reduce(0, result);
  }
};

template <LegateTypeCode CODE, int DIM>
struct ScalarUnaryRedImplBody<VariantKind::CPU, UnaryRedCode::COUNT_NONZERO, CODE, DIM> {
  using OP    = UnaryRedOp<UnaryRedCode::SUM, LegateTypeCode::UINT64_LT>;
  using LG_OP = typename OP::OP;
  using VAL   = legate_type_of<CODE>;

  void operator()(AccessorRD<LG_OP, true, 1> out,
                  AccessorRO<VAL, DIM> in,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  bool dense) const
  {
    auto result         = LG_OP::identity;
    const size_t volume = rect.volume();
    if (dense) {
      auto inptr = in.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) result += inptr[idx] != VAL(0);
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto point = pitches.unflatten(idx, rect.lo);
        result += in[point] != VAL(0);
      }
    }
    out.reduce(0, result);
  }
};

/*static*/ void ScalarUnaryRedTask::cpu_variant(TaskContext& context)
{
  scalar_unary_red_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  ScalarUnaryRedTask::register_variants();
}
}  // namespace

}  // namespace cunumeric
