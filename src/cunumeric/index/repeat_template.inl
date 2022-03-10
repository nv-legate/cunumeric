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
struct RepeatImplBody;

template <VariantKind KIND>
struct RepeatImpl {
  template <LegateTypeCode CODE, int DIM>
  void operator()(RepeatArgs& args) const
  {
    using VAL       = legate_type_of<CODE>;
    auto input_rect = args.input.shape<DIM>();
    auto input_arr  = args.input.read_accessor<VAL, DIM>(input_rect);
    Pitches<DIM - 1> input_pitches;
    Buffer<VAL> output_arr;
    size_t volume = input_pitches.flatten(input_rect);
    if (volume == 0) {
      auto empty = create_buffer<VAL>(0);
      args.output.return_data(empty, 0);
      return;
    }

    if (args.scalar_repeats) {
      auto size = RepeatImplBody<KIND, CODE, DIM>{}(
        output_arr, input_arr, args.repeats, args.axis, input_pitches, input_rect);
      args.output.return_data(output_arr, size);

    } else {
      auto r_rect      = args.repeats_arr.shape<DIM>();
      auto repeats_arr = args.repeats_arr.read_accessor<int64_t, DIM>(r_rect);
#ifdef CUNUMERIC_DEBUG
      // repeats should have the same shape and partitioning as an input array
      assert(r_rect == input_rect);
#endif
      auto size = RepeatImplBody<KIND, CODE, DIM>{}(
        output_arr, input_arr, repeats_arr, args.axis, input_pitches, input_rect);
      args.output.return_data(output_arr, size);
    }
  }
};

template <VariantKind KIND>
static void repeat_template(TaskContext& context)
{
  bool scalar_repeats = context.scalars()[1].value<bool>();
  auto axis           = context.scalars()[0].value<int32_t>();
  if (scalar_repeats) {
    auto repeats = context.scalars()[2].value<int64_t>();
    RepeatArgs args{
      context.outputs()[0], context.inputs()[0], Array(), repeats, axis, scalar_repeats};
    double_dispatch(args.input.dim(), args.input.code(), RepeatImpl<KIND>{}, args);
  } else {
    auto& repeats = context.inputs()[1];
    RepeatArgs args{context.outputs()[0], context.inputs()[0], repeats, 0, axis, scalar_repeats};
    double_dispatch(args.input.dim(), args.input.code(), RepeatImpl<KIND>{}, args);
  }
}

}  // namespace cunumeric
