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

#include "unary/scalar_unary_red.h"
#include "unary/scalar_unary_red_template.inl"

namespace legate {
namespace numpy {

using namespace Legion;

template <UnaryRedCode OP_CODE, LegateTypeCode CODE, int DIM>
struct ScalarUnaryRedImplBody<VariantKind::CPU, OP_CODE, CODE, DIM> {
  using OP  = UnaryRedOp<OP_CODE, CODE>;
  using VAL = legate_type_of<CODE>;

  void operator()(OP func,
                  VAL &result,
                  AccessorRO<VAL, DIM> in,
                  const Rect<DIM> &rect,
                  const Pitches<DIM - 1> &pitches,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    if (dense) {
      auto inptr = in.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) OP::template fold<true>(result, inptr[idx]);
    } else {
      CPULoop<DIM>::unary_reduction_loop(func, result, rect, in);
    }
  }
};

template <LegateTypeCode CODE, int DIM>
struct ScalarUnaryRedImplBody<VariantKind::CPU, UnaryRedCode::CONTAINS, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(bool &result,
                  AccessorRO<VAL, DIM> in,
                  const UntypedScalar &to_find_scalar,
                  const Rect<DIM> &rect,
                  const Pitches<DIM - 1> &pitches,
                  bool dense) const
  {
    const auto to_find  = to_find_scalar.value<VAL>();
    const size_t volume = rect.volume();
    if (dense) {
      auto inptr = in.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx)
        if (inptr[idx] == to_find) {
          result = true;
          return;
        }
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto point = pitches.unflatten(idx, rect.lo);
        if (in[point] == to_find) {
          result = true;
          return;
        }
      }
    }
  }
};

template <LegateTypeCode CODE, int DIM>
struct ScalarUnaryRedImplBody<VariantKind::CPU, UnaryRedCode::COUNT_NONZERO, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(uint64_t &result,
                  AccessorRO<VAL, DIM> in,
                  const Rect<DIM> &rect,
                  const Pitches<DIM - 1> &pitches,
                  bool dense) const
  {
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
  }
};

void deserialize(Deserializer &ctx, ScalarUnaryRedArgs &args)
{
  deserialize(ctx, args.op_code);
  deserialize(ctx, args.shape);
  deserialize(ctx, args.in);
  deserialize(ctx, args.args);
}

/*static*/ UntypedScalar ScalarUnaryRedTask::cpu_variant(const Task *task,
                                                         const std::vector<PhysicalRegion> &regions,
                                                         Context context,
                                                         Runtime *runtime)
{
  return scalar_unary_red_template<VariantKind::CPU>(task, regions, context, runtime);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  ScalarUnaryRedTask::register_variants_with_return<UntypedScalar, UntypedScalar>();
}
}  // namespace

}  // namespace numpy
}  // namespace legate
