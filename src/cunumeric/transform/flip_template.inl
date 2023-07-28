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
#include "cunumeric/transform/flip.h"
#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE, int DIM>
struct FlipImplBody;

template <VariantKind KIND>
struct FlipImpl {
  template <Type::Code CODE, int DIM>
  void operator()(FlipArgs& args) const
  {
    using VAL = legate_type_of<CODE>;

    auto rect = args.out.shape<DIM>().intersection(args.in.shape<DIM>());

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out = args.out.write_accessor<VAL, DIM>(rect);
    auto in  = args.in.read_accessor<VAL, DIM>(rect);

    FlipImplBody<KIND, CODE, DIM>()(out, in, pitches, rect, args.axes);
  }
};

template <VariantKind KIND>
static void flip_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();
  auto& scalars = context.scalars();

  FlipArgs args{inputs[0], outputs[0], scalars[0].values<int32_t>()};
  double_dispatch(args.in.dim(), args.in.code(), FlipImpl<KIND>{}, args);
}

}  // namespace cunumeric
