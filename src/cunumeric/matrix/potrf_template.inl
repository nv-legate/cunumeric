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
#include "cunumeric/matrix/potrf.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, LegateTypeCode CODE>
struct PotrfImplBody;

template <LegateTypeCode CODE>
struct support_potrf : std::false_type {};
template <>
struct support_potrf<LegateTypeCode::DOUBLE_LT> : std::true_type {};
template <>
struct support_potrf<LegateTypeCode::FLOAT_LT> : std::true_type {};
template <>
struct support_potrf<LegateTypeCode::COMPLEX64_LT> : std::true_type {};
template <>
struct support_potrf<LegateTypeCode::COMPLEX128_LT> : std::true_type {};

template <VariantKind KIND>
struct PotrfImpl {
  template <LegateTypeCode CODE, std::enable_if_t<support_potrf<CODE>::value>* = nullptr>
  void operator()(Array& array) const
  {
    using VAL = legate_type_of<CODE>;

    auto shape = array.shape<2>();

    if (shape.empty()) return;

    size_t strides[2];

    auto arr = array.write_accessor<VAL, 2>(shape).ptr(shape, strides);
    auto m   = static_cast<int32_t>(shape.hi[0] - shape.lo[0] + 1);
    auto n   = static_cast<int32_t>(shape.hi[1] - shape.lo[1] + 1);
    assert(m > 0 && n > 0);

    PotrfImplBody<KIND, CODE>()(arr, m, n);
  }

  template <LegateTypeCode CODE, std::enable_if_t<!support_potrf<CODE>::value>* = nullptr>
  void operator()(Array& array) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void potrf_template(TaskContext& context)
{
  auto& array = context.outputs()[0];
  type_dispatch(array.code(), PotrfImpl<KIND>{}, array);
}

}  // namespace cunumeric
