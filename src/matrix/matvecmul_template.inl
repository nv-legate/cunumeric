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

#include "core.h"
#include "deserializer.h"
#include "dispatch.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <VariantKind KIND, LegateTypeCode CODE>
struct MatVecMulImplBody;

template <LegateTypeCode CODE>
struct support_matvecmul : std::false_type {
};
template <>
struct support_matvecmul<LegateTypeCode::DOUBLE_LT> : std::true_type {
  using ACC_TYPE = double;
};
template <>
struct support_matvecmul<LegateTypeCode::FLOAT_LT> : std::true_type {
  using ACC_TYPE = float;
};
template <>
struct support_matvecmul<LegateTypeCode::HALF_LT> : std::true_type {
  using ACC_TYPE = float;
};

template <VariantKind KIND>
struct MatVecMulImpl {
  template <LegateTypeCode CODE, std::enable_if_t<support_matvecmul<CODE>::value> * = nullptr>
  void operator()(MatVecMulArgs &args) const
  {
    using VAL = legate_type_of<CODE>;
    using ACC = typename support_matvecmul<CODE>::ACC_TYPE;

    const bool vec_on_lhs = args.rhs1.dim() == 1;

    size_t m          = 0;
    size_t n          = 0;
    const VAL *rhs1   = nullptr;
    const VAL *rhs2   = nullptr;
    size_t rhs_stride = 0;

    auto get_dimensions = [](const Rect<2> &rect) {
      auto m = static_cast<size_t>(rect.hi[0] - rect.lo[0] + 1);
      auto n = static_cast<size_t>(rect.hi[1] - rect.lo[1] + 1);
      return std::make_pair(m, n);
    };

    auto lhs_rect = args.lhs_shape.to_rect<1>();
    if (vec_on_lhs) {
      assert(args.rhs2.dim() == 2);
      auto rhs1_rect = args.rhs1_shape.to_rect<1>();
      Rect<2> rhs2_rect(Point<2>(rhs1_rect.lo[0], lhs_rect.lo[0]),
                        Point<2>(rhs1_rect.hi[0], lhs_rect.hi[0]));

      size_t rhs1_strides[1];
      size_t rhs2_strides[2];
      rhs1           = args.rhs1.read_accessor<VAL, 1>().ptr(rhs1_rect, rhs1_strides);
      rhs2           = args.rhs2.read_accessor<VAL, 2>().ptr(rhs2_rect, rhs2_strides);
      rhs_stride     = rhs2_strides[0];
      std::tie(m, n) = get_dimensions(rhs2_rect);
    } else {
      assert(args.rhs1.dim() == 2);
      assert(args.rhs2.dim() == 1);
      auto rhs2_rect = args.rhs2_shape.to_rect<1>();
      Rect<2> rhs1_rect(Point<2>(lhs_rect.lo[0], rhs2_rect.lo[0]),
                        Point<2>(lhs_rect.hi[0], rhs2_rect.hi[0]));

      size_t rhs1_strides[2];
      size_t rhs2_strides[1];
      rhs1           = args.rhs1.read_accessor<VAL, 2>().ptr(rhs1_rect, rhs1_strides);
      rhs2           = args.rhs2.read_accessor<VAL, 1>().ptr(rhs2_rect, rhs2_strides);
      rhs_stride     = rhs1_strides[0];
      std::tie(m, n) = get_dimensions(rhs1_rect);
    }

    size_t lhs_strides[1];
    if (args.needs_reduction) {
      auto lhs = args.lhs.reduce_accessor<SumReduction<ACC>, true, 1>().ptr(lhs_rect, lhs_strides);
      MatVecMulImplBody<KIND, CODE>()(m, n, lhs, rhs1, rhs2, rhs_stride, vec_on_lhs);
    } else {
      auto lhs = args.lhs.write_accessor<VAL, 1>().ptr(lhs_rect, lhs_strides);
      MatVecMulImplBody<KIND, CODE>()(m, n, lhs, rhs1, rhs2, rhs_stride, vec_on_lhs);
    }
  }

  template <LegateTypeCode CODE, std::enable_if_t<!support_matvecmul<CODE>::value> * = nullptr>
  void operator()(MatVecMulArgs &args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void matvecmul_template(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context context,
                               Runtime *runtime)
{
  Deserializer ctx(task, regions);
  MatVecMulArgs args;
  deserialize(ctx, args);
  // Note that we can't dispatch on the lhs's type,
  // as the lhs can have a different type than the rhs'
  type_dispatch(args.rhs1.code(), MatVecMulImpl<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
