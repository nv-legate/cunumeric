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

#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, LegateTypeCode CODE, int DIM>
struct Cumsum_gImplBody;

template <VariantKind KIND>
struct Cumsum_gImpl {
  template <LegateTypeCode CODE, int DIM>
  void operator()(Cumsum_gArgs& args, const DomainPoint& partition_index) const
  {

    using VAL = legate_type_of<CODE>;
    
    auto out_rect = args.out.shape<DIM>();
    auto sum_vals_rect = args.sum_vals.shape<DIM>();

    Pitches<DIM - 1> out_pitches;
    size_t volume = out_pitches.flatten(out_rect);
    Pitches<DIM - 1> sum_vals_pitches;
    size_t sum_vals_volume = sum_vals_pitches.flatten(sum_vals_rect);

    if (volume == 0) return;

    auto out = args.out.read_write_accessor<VAL, DIM>(out_rect);
    auto sum_vals = args.sum_vals.read_accessor<VAL, DIM>(sum_vals_rect);

    Cumsum_gImplBody<KIND, CODE, DIM>()(out, sum_vals, out_pitches, out_rect, sum_vals_pitches, sum_vals_rect, partition_index);

  }
};

template <VariantKind KIND>
static void Cumsum_g_template(TaskContext& context)
{
  Cumsum_gArgs args{context.inputs()[1], context.outputs()[0]};
  double_dispatch(args.out.dim(), args.out.code(), Cumsum_gImpl<KIND>{}, args, context.get_task_index());  
}

}  // namespace cunumeric
