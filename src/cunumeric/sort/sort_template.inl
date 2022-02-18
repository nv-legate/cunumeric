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

template <VariantKind KIND, bool ARGSORT, LegateTypeCode CODE, int32_t DIM>
struct SortImplBody;

static int getRank(Domain domain, DomainPoint index_point)
{
  int domain_index = 0;
  for (int i = 0; i < domain.get_dim(); ++i) {
    if (i > 0) domain_index *= domain.hi()[i] - domain.lo()[i] + 1;
    domain_index += index_point[i];
  }
  return domain_index;
}

template <VariantKind KIND>
struct SortImpl {
  template <LegateTypeCode CODE, int32_t DIM>
  void operator()(SortArgs& args) const
  {
    using VAL = legate_type_of<CODE>;

    auto rect = args.input.shape<DIM>();

    // we shall not return on empty rectangle in case of distributed data
    // as the process might still participate in the parallel sort
    if ((DIM > 1 || !args.is_index_space) && rect.empty()) return;

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    /*
     * Assumptions:
     * 1. Sort is always requested for the 'last' dimension within rect
     * 2. We have product_of_all_other_dimensions independent sort ranges
     * 3. if we have more than one participants:
     *  a) 1D-case: we need to perform parallel sort (e.g. via sampling) -- not implemented yet
     *  b) ND-case: rect needs to be the full domain in that last dimension
     *
     */

#ifdef DEBUG_CUNUMERIC
    std::cout << "DIM=" << DIM << ", rect=" << rect << ", shape=" << args.global_shape
              << ", argsort=" << args.argsort << ", sort_dim_size=" << args.global_shape[DIM - 1]
              << std::endl;

    assert((DIM == 1 || (rect.hi[DIM - 1] - rect.lo[DIM - 1] + 1 == args.global_shape[DIM - 1])) &&
           "multi-dimensional array should not be distributed in (sort) dimension");
#endif

    auto input = args.input.read_accessor<VAL, DIM>(rect);

    if (args.argsort) {
      auto output = args.output.write_accessor<int32_t, DIM>(rect);

#ifndef LEGION_BOUNDS_CHECKS
      bool dense =
        input.accessor.is_dense_row_major(rect) && output.accessor.is_dense_row_major(rect);
#else
      bool dense = false;
#endif
      assert(dense || !args.is_index_space || DIM > 1);

      SortImplBody<KIND, true, CODE, DIM>()(input,
                                            output,
                                            pitches,
                                            rect,
                                            dense,
                                            volume,
                                            args.argsort,
                                            args.global_shape,
                                            args.is_index_space,
                                            args.task_index,
                                            args.launch_domain);

    } else {
      auto output = args.output.write_accessor<VAL, DIM>(rect);

#ifndef LEGION_BOUNDS_CHECKS
      bool dense =
        input.accessor.is_dense_row_major(rect) && output.accessor.is_dense_row_major(rect);
#else
      bool dense = false;
#endif
      assert(dense || !args.is_index_space || DIM > 1);
      SortImplBody<KIND, false, CODE, DIM>()(input,
                                             output,
                                             pitches,
                                             rect,
                                             dense,
                                             volume,
                                             args.argsort,
                                             args.global_shape,
                                             args.is_index_space,
                                             args.task_index,
                                             args.launch_domain);
    }
  }
};

template <VariantKind KIND>
static void sort_template(TaskContext& context)
{
  DomainPoint global_shape;
  {
    auto shape_span  = context.scalars()[1].values<int32_t>();
    global_shape.dim = shape_span.size();
    for (int32_t dim = 0; dim < global_shape.dim; ++dim) { global_shape[dim] = shape_span[dim]; }
  }

  SortArgs args{context.inputs()[0],
                context.outputs()[0],
                context.scalars()[0].value<bool>(),
                global_shape,
                !context.is_single_task(),
                context.get_task_index(),
                context.get_launch_domain()};
  double_dispatch(args.input.dim(), args.input.code(), SortImpl<KIND>{}, args);
}

}  // namespace cunumeric
