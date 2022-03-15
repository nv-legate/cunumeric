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

template <VariantKind KIND, LegateTypeCode CODE, int DIM1, int DIM2>
struct AdvancedIndexingImplBody;

template <VariantKind KIND, LegateTypeCode CODE, int DIM1>
struct AdvancedIndexingImpl {
  template <int DIM2>
  void operator()(AdvancedIndexingArgs& args) const
  {
    using VAL       = legate_type_of<CODE>;
    auto input_rect = args.input_array.shape<DIM1>();
    auto input_arr  = args.input_array.read_accessor<VAL, DIM1>(input_rect);
    Pitches<DIM1 - 1> input_pitches;
    Buffer<VAL> output_arr;
    size_t volume1 = input_pitches.flatten(input_rect);

    auto index_rect = args.indexing_array.shape<DIM2>();
    auto index_arr  = args.indexing_array.read_accessor<bool, DIM2>(index_rect);
    Pitches<DIM2 - 1> index_pitches;
    size_t volume2 = index_pitches.flatten(index_rect);

    if (volume1 == 0 || volume2 == 0) {
      auto empty = create_buffer<VAL>(0);
      args.output.return_data(empty, 0);
      return;
    }

    int64_t size = 0;
    if (DIM1 == DIM2) {
      size = AdvancedIndexingImplBody<KIND, CODE, DIM1, DIM2>{}(
        output_arr, input_arr, index_arr, input_pitches, input_rect, index_pitches, index_rect);
    } else {
      // should never go here, not implemented
      assert(false);
    }
    args.output.return_data(output_arr, size);
  }
};

template <VariantKind KIND>
struct AdvancedIndexingHelper {
  template <LegateTypeCode CODE, int DIM1>
  void operator()(AdvancedIndexingArgs& args) const
  {
    dim_dispatch(args.indexing_array.dim(), AdvancedIndexingImpl<KIND, CODE, DIM1>{}, args);
  }
};

template <VariantKind KIND>
static void advanced_indexing_template(TaskContext& context)
{
  AdvancedIndexingArgs args{context.outputs()[0], context.inputs()[0], context.inputs()[1]};
  double_dispatch(
    args.input_array.dim(), args.input_array.code(), AdvancedIndexingHelper<KIND>{}, args);
}

}  // namespace cunumeric
