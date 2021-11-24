/* Copyright 2021 NVIDIA Corporation
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
struct ChooseImplBody;

template <VariantKind KIND>
struct ChooseImpl {
  template <LegateTypeCode CODE, int DIM>
  void operator()(ChooseArgs& args) const
  {
    using VAL = legate_type_of<CODE>;
    auto rect = args.out.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);
    if (volume == 0) return;

    auto out       = args.out.write_accessor<VAL, DIM>(rect);
    auto index_arr = args.inputs[0].read_accessor<int, DIM>(rect);

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense =
      index_arr.accessor.is_dense_row_major(rect) && out.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif
    std::vector<AccessorRO<VAL, DIM>> choices;
    auto rect_c = args.inputs[1].shape<DIM>();
    for (int i = 1; i < args.inputs.size(); i++) {
      choices.push_back(args.inputs[i].read_accessor<VAL, DIM>(rect_c));
      dense = dense && choices[i - 1].accessor.is_dense_row_major(rect_c);
    }
    ChooseImplBody<KIND, CODE, DIM>()(out, index_arr, choices, rect, pitches, dense);
  }
};

template <VariantKind KIND>
static void choose_template(TaskContext& context)
{
  ChooseArgs args{context.outputs()[0], context.inputs()};
  double_dispatch(args.inputs[0].dim(), args.inputs[0].code(), ChooseImpl<KIND>{}, args);
}

}  // namespace cunumeric
