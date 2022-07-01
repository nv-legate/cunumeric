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
#include "cunumeric/sort/searchsorted.h"
#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, LegateTypeCode CODE>
struct SearchSortedImplBody;

template <VariantKind KIND>
struct SearchSortedImpl {
  template <LegateTypeCode CODE>
  void operator()(SearchSortedArgs& args) const
  {
    using VAL = legate_type_of<CODE>;

    auto rect_base       = args.input_base.shape<1>();
    auto rect_values_in  = args.input_values.shape<1>();
    auto rect_values_out = args.output_reduction.shape<1>();

    if (rect_base.empty()) return;

    Pitches<0> pitches;
    size_t volume     = pitches.flatten(rect_base);
    size_t num_values = pitches.flatten(rect_values_in);
    assert(num_values == pitches.flatten(rect_values_out));

    SearchSortedImplBody<KIND, CODE>()(args.input_base,
                                       args.input_values,
                                       args.output_reduction,
                                       rect_base,
                                       rect_values_in,
                                       args.left,
                                       args.is_index_space,
                                       volume,
                                       args.global_volume,
                                       num_values);
  }
};

template <VariantKind KIND>
static void searchsorted_template(TaskContext& context)
{
  SearchSortedArgs args{context.inputs()[0],
                        context.inputs()[1],
                        context.reductions()[0],
                        context.scalars()[0].value<bool>(),
                        context.scalars()[1].value<int64_t>(),
                        !context.is_single_task()};

  assert(args.input_base.dim() == 1);
  assert(args.input_base.code() == args.input_values.code());
  assert(args.input_values.dim() == args.output_reduction.dim());

  type_dispatch(args.input_base.code(), SearchSortedImpl<KIND>{}, args);
}

}  // namespace cunumeric
