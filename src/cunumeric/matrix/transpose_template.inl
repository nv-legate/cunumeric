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
#include "cunumeric/matrix/transpose.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct TransposeImplBody;

template <VariantKind KIND>
struct TransposeImpl {
  template <Type::Code CODE>
  void operator()(TransposeArgs& args) const
  {
    using VAL = legate_type_of<CODE>;

    const auto out_rect = args.out.shape<2>();
    if (out_rect.empty()) return;

    Rect<2> in_rect;
    if (args.logical) {
      in_rect.lo = Point<2>(out_rect.lo[1], out_rect.lo[0]);
      in_rect.hi = Point<2>(out_rect.hi[1], out_rect.hi[0]);
    } else
      in_rect = out_rect;

    auto out = args.out.write_accessor<VAL, 2>();
    auto in  = args.in.read_accessor<VAL, 2>();

    TransposeImplBody<KIND, CODE>{}(out_rect, in_rect, out, in, args.logical);
  }
};

template <VariantKind KIND>
static void transpose_template(TaskContext& context)
{
  auto& output = context.outputs()[0];
  auto& input  = context.inputs()[0];
  auto logical = context.scalars()[0].value<bool>();

  TransposeArgs args{output, input, logical};
  type_dispatch(input.code(), TransposeImpl<KIND>{}, args);
}

}  // namespace cunumeric
