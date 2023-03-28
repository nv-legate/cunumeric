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

// Useful for IDEs
#include <core/utilities/typedefs.h>
#include "cunumeric/unary/isnan.h"
#include "cunumeric/unary/nanargmax.h"
#include "cunumeric/pitches.h"
#include "cunumeric/execution_policy/indexing/parallel_loop.h"

// TODO: Eventually, this should use an execution policy for reductions
// and not that of indexing, but since there are no reductions right now,
// execution policy for indexing is used as a placeholder

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, LegateTypeCode CODE, int DIM>
struct NanArgMax {
  using T  = legate_type_of<CODE>;
  using IN = AccessorRW<T, DIM>;

  IN input;
  Pitches<DIM - 1> pitches;
  Rect<DIM> rect;
  size_t volume;
  float identity;

  struct SampleTag {};

  // constructor:
  NanArgMax(NanArgMaxArgs& args)
  {
    rect = args.input.shape<DIM>();

    identity = args.identity;

    input  = args.input.read_write_accessor<T, DIM>(rect);
    volume = pitches.flatten(rect);
    if (volume == 0) return;
  }

  __CUDA_HD__ void operator()(const size_t idx, SampleTag) const noexcept
  {
    // auto p = pitches.unflatten(idx, rect.lo);
    // input[p] = input[p];

    auto inptr = input.ptr(rect);

    if (is_nan(inptr[idx])) {
      inptr[idx] = identity;
    } else {
      inptr[idx] = inptr[idx];
    }
  }

  void execute() const noexcept { return ParallelLoopPolicy<KIND, SampleTag>()(rect, *this); }
};

using namespace legate;

template <VariantKind KIND>
struct NanArgMaxImpl {
  template <LegateTypeCode CODE, int DIM>
  void operator()(NanArgMaxArgs& args) const
  {
    NanArgMax<KIND, CODE, DIM> nanargmax(args);
    nanargmax.execute();
  }
};

template <VariantKind KIND>
static void nanargmax_template(TaskContext& context)
{
  auto& inputs = context.inputs();
  NanArgMaxArgs args{context.outputs()[0], context.scalars()[0].value<float>()};
  double_dispatch(args.input.dim(), args.input.code(), NanArgMaxImpl<KIND>{}, args);
}

}  // namespace cunumeric
