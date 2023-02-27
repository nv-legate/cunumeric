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
#include "cunumeric/nullary/arange.h"
#include "cunumeric/arg.h"
#include "cunumeric/arg.inl"
#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, typename VAL>
struct ArangeImplBody;

template <VariantKind KIND>
struct ArangeImpl {
  template <LegateTypeCode CODE>
  void operator()(ArangeArgs& args) const
  {
    using VAL = legate_type_of<CODE>;

    const auto rect = args.out.shape<1>();

    if (rect.empty()) return;

    auto out = args.out.write_accessor<VAL, 1>();

    const auto start = args.start.scalar<VAL>();
    const auto step  = args.step.scalar<VAL>();

    ArangeImplBody<KIND, VAL>{}(out, rect, start, step);
  }
};

template <VariantKind KIND>
static void arange_template(TaskContext& context)
{
  auto& inputs = context.inputs();
  ArangeArgs args{context.outputs()[0], inputs[0], inputs[1], inputs[2]};
  type_dispatch(args.out.code(), ArangeImpl<KIND>{}, args);
}

}  // namespace cunumeric
