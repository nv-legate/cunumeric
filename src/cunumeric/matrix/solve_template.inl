/* Copyright 2022 NVIDIA Corporation
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

#include <vector>

#include "core/comm/coll.h"

// Useful for IDEs
#include "cunumeric/matrix/solve.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, LegateTypeCode CODE>
struct SolveImplBody;

template <LegateTypeCode CODE>
struct support_solve : std::false_type {
};
template <>
struct support_solve<LegateTypeCode::DOUBLE_LT> : std::true_type {
};
template <>
struct support_solve<LegateTypeCode::FLOAT_LT> : std::true_type {
};
template <>
struct support_solve<LegateTypeCode::COMPLEX64_LT> : std::true_type {
};
template <>
struct support_solve<LegateTypeCode::COMPLEX128_LT> : std::true_type {
};

template <VariantKind KIND>
struct SolveImpl {
  template <LegateTypeCode CODE, std::enable_if_t<support_solve<CODE>::value>* = nullptr>
  void operator()(Array& a_array, Array& b_array) const
  {
    using VAL = legate_type_of<CODE>;

    const auto a_shape = a_array.shape<2>();
    const auto b_shape = b_array.shape<2>();

    if (a_shape.empty() || b_shape.empty()) return;

    size_t a_strides[2];
    size_t b_strides[2];

    const auto m    = static_cast<int64_t>(a_shape.hi[0] - a_shape.lo[0] + 1);
    const auto n    = static_cast<int64_t>(a_shape.hi[1] - a_shape.lo[1] + 1);
    const auto nrhs = static_cast<int64_t>(b_shape.hi[1] - b_shape.lo[1] + 1);
    auto a          = a_array.write_accessor<VAL, 2>(a_shape).ptr(a_shape, a_strides);
    auto b          = b_array.write_accessor<VAL, 2>(b_shape).ptr(b_shape, b_strides);

    assert(m > 0 && n > 0 && nrhs > 0);

    SolveImplBody<KIND, CODE>()(m, n, nrhs, a, b);
  }

  template <LegateTypeCode CODE, std::enable_if_t<!support_solve<CODE>::value>* = nullptr>
  void operator()(Array& a_array, Array& b_array) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void solve_template(TaskContext& context)
{
  auto& a_array = context.outputs()[0];
  auto& b_array = context.outputs()[1];
  type_dispatch(a_array.code(), SolveImpl<KIND>{}, a_array, b_array);
}

}  // namespace cunumeric
