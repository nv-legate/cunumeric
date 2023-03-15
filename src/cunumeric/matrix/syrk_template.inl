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

#pragma once

// Useful for IDEs
#include "cunumeric/matrix/syrk.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, LegateTypeCode CODE>
struct SyrkImplBody;

template <LegateTypeCode CODE>
struct support_syrk : std::false_type {};
template <>
struct support_syrk<LegateTypeCode::DOUBLE_LT> : std::true_type {};
template <>
struct support_syrk<LegateTypeCode::FLOAT_LT> : std::true_type {};
template <>
struct support_syrk<LegateTypeCode::COMPLEX64_LT> : std::true_type {};
template <>
struct support_syrk<LegateTypeCode::COMPLEX128_LT> : std::true_type {};

template <VariantKind KIND>
struct SyrkImpl {
  template <LegateTypeCode CODE, std::enable_if_t<support_syrk<CODE>::value>* = nullptr>
  void operator()(Array& lhs_array, Array& rhs_array) const
  {
    using VAL = legate_type_of<CODE>;

    auto lhs_shape = lhs_array.shape<2>();
    auto rhs_shape = rhs_array.shape<2>();

    if (lhs_shape.empty()) return;

    size_t lhs_strides[2];
    size_t rhs_strides[2];

    auto lhs = lhs_array.write_accessor<VAL, 2>(lhs_shape).ptr(lhs_shape, lhs_strides);
    auto rhs = rhs_array.read_accessor<VAL, 2>(rhs_shape).ptr(rhs_shape, rhs_strides);

    auto m = static_cast<int32_t>(rhs_shape.hi[0] - rhs_shape.lo[0] + 1);
    auto n = static_cast<int32_t>(rhs_shape.hi[1] - rhs_shape.lo[1] + 1);
    assert(m > 0 && n > 0);

    SyrkImplBody<KIND, CODE>()(lhs, rhs, m, n);
  }

  template <LegateTypeCode CODE, std::enable_if_t<!support_syrk<CODE>::value>* = nullptr>
  void operator()(Array& lhs_array, Array& rhs_array) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void syrk_template(TaskContext& context)
{
  auto& lhs = context.outputs()[0];
  auto& rhs = context.inputs()[0];

  type_dispatch(lhs.code(), SyrkImpl<KIND>{}, lhs, rhs);
}

}  // namespace cunumeric
