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

#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, LegateTypeCode CODE, int DIM>
struct CopyImplBody;

template <VariantKind KIND>
struct CopyImpl {
  template <LegateTypeCode CODE, int DIM>
  void operator()(CopyArgs& args) const
  {
    using VAL   = legate_type_of<CODE>;
    using POINT = Point<DIM>;

    auto rect_out = args.out.shape<DIM>();
    auto rect_in  = args.in[0].shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect_out);

    if (volume == 0) return;

    auto out = args.out.write_accessor<VAL, DIM>(rect_out);
    auto in  = args.in[0].read_accessor<VAL, DIM>(rect_in);

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense =
      out.accessor.is_dense_row_major(rect_out) && in.accessor.is_dense_row_major(rect_out);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif
    if (args.in.size() == 2) {
      auto rect_indirect = args.in[1].shape<DIM>();
      Pitches<DIM - 1> pitches_indirect;
      size_t volume_ind = pitches.flatten(rect_indirect);
      if (volume_ind == 0) return;
      auto indirection = args.in[1].read_accessor<POINT, DIM>(rect_indirect);

#ifndef LEGION_BOUNDS_CHECKS
      // Check to see if this is dense or not
      dense = dense && indirection.accessor.is_dense_row_major(rect_out);
#endif
#ifdef DEBUG_CUNUMERIC
      if (args.is_source_indirect)
        assert(rect_out == rect_indirect);
      else
        assert(rect_in == rect_indirect);

      if (dense) assert(rect_in == rect_out);
#endif

      CopyImplBody<KIND, CODE, DIM>()(
        out, in, indirection, pitches_indirect, rect_indirect, dense, args.is_source_indirect);
    } else {  // no indirection
#ifdef DEBUG_CUNUMERIC
      assert(rect_in == rect_out);
#endif
      CopyImplBody<KIND, CODE, DIM>()(out, in, pitches, rect_out, dense);
    }
  }
};

template <VariantKind KIND>
static void copy_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();
  auto& scalars = context.scalars();

  bool is_source_indirect = false;
  if (scalars.size() >= 1) is_source_indirect = scalars[0].value<bool>();

  CopyArgs args{inputs, outputs[0], is_source_indirect};
  auto dim = args.in[0].dim();
  double_dispatch(dim, args.in[0].code(), CopyImpl<KIND>{}, args);
}

}  // namespace cunumeric
