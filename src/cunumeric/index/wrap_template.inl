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
#include "cunumeric/index/wrap.h"
#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, int DIM>
struct WrapImplBody;

template <VariantKind KIND>
struct WrapImpl {
  template <int DIM>
  void operator()(WrapArgs& args) const
  {
    using VAL     = Point<DIM>;
    auto out_rect = args.out.shape<1>();  // output array is always 1D
    auto out      = args.out.write_accessor<Point<DIM>, 1>(out_rect);

    Pitches<0> pitches_out;
    size_t volume_out = pitches_out.flatten(out_rect);
    if (volume_out == 0) return;

#ifndef LEGION_BOUNDS_CHECKS
    bool dense = out.accessor.is_dense_row_major(out_rect);
#else
    bool dense = false;
#endif

    Point<DIM> point_lo, point_hi;
    for (int dim = 0; dim < DIM; ++dim) {
      point_lo[dim] = 0;
      point_hi[dim] = args.shape[dim] - 1;
    }
    Rect<DIM> input_rect(point_lo, point_hi);

    Pitches<DIM - 1> pitches_in;
    size_t volume_in = pitches_in.flatten(input_rect);
#ifdef DEBUG_CUNUMERIC
    assert(volume_in != 0);
#endif

    if (args.has_input) {
      auto in_rect = args.in.shape<1>();
      auto in = args.in.read_accessor<int64_t, 1>(in_rect);  // input should be always integer type
#ifdef DEBUG_CUNUMERIC
      assert(in_rect == out_rect);
#endif
      WrapImplBody<KIND, DIM>()(
        out, pitches_out, out_rect, pitches_in, input_rect, dense, args.check_bounds, in);

    } else {
      bool tmp = false;
      WrapImplBody<KIND, DIM>()(
        out, pitches_out, out_rect, pitches_in, input_rect, dense, args.check_bounds, tmp);
    }  // else
  }
};

template <VariantKind KIND>
static void wrap_template(TaskContext& context)
{
  auto shape        = context.scalars()[0].value<DomainPoint>();
  int dim           = shape.dim;
  bool has_input    = context.scalars()[1].value<bool>();
  bool check_bounds = context.scalars()[2].value<bool>();
  Array tmp_array   = Array();
  WrapArgs args{context.outputs()[0],
                shape,
                has_input,
                check_bounds,
                has_input ? context.inputs()[0] : tmp_array};
  dim_dispatch(dim, WrapImpl<KIND>{}, args);
}

}  // namespace cunumeric
