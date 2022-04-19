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
struct ScanLocalImplBody;

template <VariantKind KIND>
struct ScanLocalImpl {
  template <LegateTypeCode CODE, int DIM>
  void operator()(ScanLocalArgs& args) const
  {
    using VAL = legate_type_of<CODE>;
    
    auto rect = args.out.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out = args.out.write_accessor<VAL, DIM>(rect);
    auto in = args.in.read_accessor<VAL, DIM>(rect);

    ScanLocalImplBody<KIND, CODE, DIM>()(out, in, args.sum_vals, pitches, rect, args.prod);

  }
};

template <VariantKind KIND>
static void scan_local_template(TaskContext& context)
{
  ScanLocalArgs args{context.outputs()[0], context.inputs()[0], context.outputs()[1]};
  double_dispatch(args.in.dim(), args.in.code(), ScanLocalImpl<KIND>{}, args);
}

}  // namespace cunumeric
