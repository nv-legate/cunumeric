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
#include "cunumeric/nullary/fill.h"
#include "cunumeric/arg.h"
#include "cunumeric/arg.inl"
#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, typename VAL, int DIM>
struct FillImplBody;

template <VariantKind KIND>
struct FillImpl {
  template <typename VAL, int DIM>
  void fill(FillArgs& args) const
  {
    auto rect = args.out.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out = args.out.write_accessor<VAL, DIM>(rect);
#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    auto fill_value = args.fill_value.read_accessor<VAL, 1>();
    FillImplBody<KIND, VAL, DIM>{}(out, fill_value, pitches, rect, dense);
  }
  template <typename VAL, int DIM>
  void fill_zero(FillArgs& args) const
  {
    auto rect = args.out.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out = args.out.write_accessor<VAL, DIM>(rect);
#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    FillImplBody<KIND, VAL, DIM>{}(out, pitches, rect, dense);
  }

  template <LegateTypeCode CODE, int DIM>
  void operator()(FillArgs& args) const
  {
    if (args.is_argval) {
      using VAL = Argval<legate_type_of<CODE>>;
      assert(args.is_zero == false);
      fill<VAL, DIM>(args);
    } else {
      using VAL = legate_type_of<CODE>;
      if (args.is_zero)
        fill_zero<VAL, DIM>(args);
      else
        fill<VAL, DIM>(args);
    }
  }
};

template <VariantKind KIND>
static void fill_template(TaskContext& context)
{
  bool is_zero = context.scalars()[1].value<bool>();
  if (is_zero) {
    FillArgs args{context.outputs()[0], Array(), context.scalars()[0].value<bool>(), true};
    double_dispatch(args.out.dim(), args.out.code(), FillImpl<KIND>{}, args);
  } else {
    FillArgs args{
      context.outputs()[0], context.inputs()[0], context.scalars()[0].value<bool>(), false};
    double_dispatch(args.out.dim(), args.out.code(), FillImpl<KIND>{}, args);
  }
}

}  // namespace cunumeric
