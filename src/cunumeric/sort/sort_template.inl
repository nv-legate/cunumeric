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

template <VariantKind KIND, LegateTypeCode CODE, int32_t DIM>
struct SortImplBody;

template <VariantKind KIND>
struct SortImpl {
  template <LegateTypeCode CODE, int32_t DIM>
  void operator()(SortArgs& args) const
  {
    using VAL = legate_type_of<CODE>;

    auto rect = args.output.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    auto inout = args.output.read_write_accessor<VAL, DIM>(rect);

    /*
     * Assumptions:
     * 1. Sort is always requested for the 'last' dimension within rect
     * 2. We have product_of_all_other_dimensions independent sort ranges
     * 3. if we have more than one participants:
     *  a) 1D-case: we need to perform parallel sort (e.g. via sampling)
     *  b) ND-case: rect needs to be the full domain in that last dimension
     */

#ifdef DEBUG_CUNUMERIC
    std::cout << "DIM=" << DIM << ", rect=" << rect << ", sort_dim_size=" << args.sort_dim_size
              << std::endl;

    assert((DIM == 1 || (rect.hi[DIM - 1] - rect.lo[DIM - 1] + 1 == args.sort_dim_size)) &&
           "multi-dimensional array should not be distributed in last (sort) dimension");
#endif

    SortImplBody<KIND, CODE, DIM>()(inout.ptr(rect),
                                    pitches,
                                    rect,
                                    volume,
                                    args.sort_dim_size,
                                    args.is_index_space,
                                    args.index_point,
                                    args.domain);
  }
};

template <VariantKind KIND>
static void sort_template(TaskContext& context)
{
  SortArgs args{context.outputs()[0],
                context.scalars()[0].value<size_t>(),
                context.task_->is_index_space,
                context.task_->index_point,
                context.task_->index_domain};
  double_dispatch(args.output.dim(), args.output.code(), SortImpl<KIND>{}, args);
}

}  // namespace cunumeric
