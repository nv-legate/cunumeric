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
struct MatMulImplBody;

template <LegateTypeCode CODE>
struct support_matmul : std::false_type {
};
template <>
struct support_matmul<LegateTypeCode::DOUBLE_LT> : std::true_type {
  using ACC_TYPE = double;
};
template <>
struct support_matmul<LegateTypeCode::FLOAT_LT> : std::true_type {
  using ACC_TYPE = float;
};
template <>
struct support_matmul<LegateTypeCode::HALF_LT> : std::true_type {
  using ACC_TYPE = float;
};

template <VariantKind KIND>
struct MatMulImpl {
  template <LegateTypeCode CODE, std::enable_if_t<support_matmul<CODE>::value> * = nullptr>
  void operator()(MatMulArgs &args) const
  {
    using VAL = legate_type_of<CODE>;
    using ACC = typename support_matmul<CODE>::ACC_TYPE;

    assert(args.lhs_shape.dim() == 2);
    assert(args.rhs1_shape.dim() == 2);
    assert(args.rhs2_shape.dim() == 2);
    assert(args.lhs.dim() == 2);
    assert(args.rhs1.dim() == 2);
    assert(args.rhs2.dim() == 2);

    auto lhs_rect  = args.lhs_shape.to_rect<2>();
    auto rhs1_rect = args.rhs1_shape.to_rect<2>();
    auto rhs2_rect = args.rhs2_shape.to_rect<2>();

    const auto m = lhs_rect.hi[0] - lhs_rect.lo[0] + 1;
    const auto n = lhs_rect.hi[1] - lhs_rect.lo[1] + 1;
    const auto k = rhs2_rect.hi[0] - rhs2_rect.lo[0] + 1;

    size_t lhs_strides[2];
    size_t rhs1_strides[2];
    size_t rhs2_strides[2];

    auto rhs1 = args.rhs1.read_accessor<VAL, 2>().ptr(rhs1_rect, rhs1_strides);
    auto rhs2 = args.rhs2.read_accessor<VAL, 2>().ptr(rhs2_rect, rhs2_strides);
    if (args.needs_reduction) {
      auto lhs = args.lhs.reduce_accessor<SumReduction<ACC>, true, 2>().ptr(lhs_rect, lhs_strides);
      MatMulImplBody<KIND, CODE>()(
        m, n, k, lhs, rhs1, rhs2, lhs_strides[0], rhs1_strides[0], rhs2_strides[0]);
    } else {
      auto lhs = args.lhs.write_accessor<VAL, 2>().ptr(lhs_rect, lhs_strides);
      MatMulImplBody<KIND, CODE>()(
        m, n, k, lhs, rhs1, rhs2, lhs_strides[0], rhs1_strides[0], rhs2_strides[0]);
    }
  }

  template <LegateTypeCode CODE, std::enable_if_t<!support_matmul<CODE>::value> * = nullptr>
  void operator()(MatMulArgs &args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void matmul_template(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context context,
                            Runtime *runtime)
{
  Deserializer ctx(task, regions);
  MatMulArgs args;
  deserialize(ctx, args);
  // Note that we can't dispatch on the lhs's type,
  // as the lhs can have a different type than the rhs'
  type_dispatch(args.rhs1.code(), MatMulImpl<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
