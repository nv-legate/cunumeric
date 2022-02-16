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

    // we shall not return on empty rectangle in case of distributed data
    // as the process might still participate in the parallel sort
    if ((DIM > 1 || !args.is_index_space) && rect.empty()) return;

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
     *
     *  FIXME: understand legion-dim != ndarray-dim case
     *
     *
     */

#ifdef DEBUG_CUNUMERIC
    std::cout << "DIM=" << DIM << ", rect=" << rect << ", shape=" << args.global_shape
              << ", axis=" << args.sort_axis
              << ", sort_dim_size=" << args.global_shape[args.sort_axis] << std::endl;

    assert((DIM == 1 || (rect.hi[args.sort_axis] - rect.lo[args.sort_axis] + 1 ==
                         args.global_shape[args.sort_axis])) &&
           "multi-dimensional array should not be distributed in (sort) dimension");
#endif

#ifndef LEGION_BOUNDS_CHECKS
    bool dense = inout.accessor.is_dense_row_major(rect);
#else
    bool dense = false;
#endif

    if (dense) {
      SortImplBody<KIND, CODE, DIM>()(inout.ptr(rect),
                                      pitches,
                                      rect,
                                      volume,
                                      args.sort_axis,
                                      args.global_shape,
                                      args.is_index_space,
                                      args.index_point,
                                      args.domain);
    } else {
      // NOTE: we might want to place this loop logic in the different KIND-implementations in
      // norder to re-use buffers

      assert(!args.is_index_space || DIM > 1);
      // compute contiguous memory block
      int contiguous_elements = 1;
      for (int i = DIM - 1; i >= 0; i--) {
        auto diff = 1 + rect.hi[i] - rect.lo[i];
        contiguous_elements *= diff;
        if (diff < args.global_shape[i]) { break; }
      }

      uint64_t elements_processed = 0;
      while (elements_processed < volume) {
        Legion::Point<DIM> start_point = pitches.unflatten(elements_processed, rect.lo);
        // RUN based on current start point
        SortImplBody<KIND, CODE, DIM>()(&(inout[start_point]),
                                        pitches,
                                        rect,
                                        contiguous_elements,
                                        args.sort_axis,
                                        args.global_shape,
                                        args.is_index_space,
                                        args.index_point,
                                        args.domain);
        elements_processed += contiguous_elements;
      }
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

  SortArgs args{context.outputs()[0],
                context.scalars()[0].value<uint32_t>(),
                global_shape,
                context.task_->is_index_space,
                context.task_->index_point,
                context.task_->index_domain};
  double_dispatch(args.output.dim(), args.output.code(), SortImpl<KIND>{}, args);
}

}  // namespace cunumeric
