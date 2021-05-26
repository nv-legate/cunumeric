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
#include "scalar.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <VariantKind KIND, LegateTypeCode CODE>
struct DotImplBody;

template <VariantKind KIND>
struct DotImpl {
  template <LegateTypeCode CODE>
  UntypedScalar operator()(DotArgs &args) const
  {
    using VAL = legate_type_of<CODE>;

    assert(args.shape.dim() == 1);
    assert(args.rhs1.dim() == 1);
    assert(args.rhs2.dim() == 1);

    auto rect  = args.shape.to_rect<1>();
    VAL result = SumReduction<VAL>::identity;

    if (rect.empty()) return UntypedScalar(result);

    auto rhs1 = args.rhs1.read_accessor<VAL, 1>();
    auto rhs2 = args.rhs2.read_accessor<VAL, 1>();

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = rhs1.accessor.is_dense_row_major(rect) && rhs2.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    DotImplBody<KIND, CODE>()(result, rhs1, rhs2, rect, dense);
    return UntypedScalar(result);
  }
};

template <VariantKind KIND>
static UntypedScalar dot_template(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context context,
                                  Runtime *runtime)
{
  Deserializer ctx(task, regions);
  DotArgs args;
  deserialize(ctx, args);
  return type_dispatch(args.rhs1.code(), DotImpl<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
