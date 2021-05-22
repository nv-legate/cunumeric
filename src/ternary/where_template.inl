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

#include "core.h"
#include "deserializer.h"
#include "dispatch.h"
#include "point_task.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <VariantKind KIND, LegateTypeCode CODE, int DIM>
struct WhereImplBody;

template <VariantKind KIND>
struct WhereImpl {
  template <LegateTypeCode CODE, int DIM>
  void operator()(WhereArgs &args) const
  {
    using VAL = legate_type_of<CODE>;

    auto rect = args.shape.to_rect<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out  = args.out.write_accessor<VAL, DIM>();
    auto mask = args.mask.read_accessor<bool, DIM>();
    auto in1  = args.in1.read_accessor<VAL, DIM>();
    auto in2  = args.in2.read_accessor<VAL, DIM>();

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in1.accessor.is_dense_row_major(rect) &&
                 in2.accessor.is_dense_row_major(rect) && mask.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    WhereImplBody<KIND, CODE, DIM>()(out, mask, in1, in2, pitches, rect, dense);
  }
};

template <VariantKind KIND>
static void where_template(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context context,
                           Runtime *runtime)
{
  Deserializer ctx(task, regions);
  WhereArgs args;
  deserialize(ctx, args);
  double_dispatch(args.out.dim(), args.out.code(), WhereImpl<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
