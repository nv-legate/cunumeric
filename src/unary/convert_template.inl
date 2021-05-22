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
#include "unary/convert_util.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <VariantKind KIND, LegateTypeCode DST_TYPE, LegateTypeCode SRC_TYPE, int DIM>
struct ConvertImplBody;

template <VariantKind KIND, LegateTypeCode SRC_TYPE>
struct ConvertImpl {
  template <LegateTypeCode DST_TYPE, int DIM, std::enable_if_t<SRC_TYPE != DST_TYPE> * = nullptr>
  void operator()(ConvertArgs &args) const
  {
    using OP  = ConvertOp<DST_TYPE, SRC_TYPE>;
    using SRC = legate_type_of<SRC_TYPE>;
    using DST = legate_type_of<DST_TYPE>;

    auto rect = args.shape.to_rect<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out = args.out.write_accessor<DST, DIM>();
    auto in  = args.in.read_accessor<SRC, DIM>();

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    OP func{};
    ConvertImplBody<KIND, DST_TYPE, SRC_TYPE, DIM>()(func, out, in, pitches, rect, dense);
  }

  template <LegateTypeCode DST_TYPE, int DIM, std::enable_if_t<SRC_TYPE == DST_TYPE> * = nullptr>
  void operator()(ConvertArgs &args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct SourceTypeDispatch {
  template <LegateTypeCode SRC_TYPE>
  void operator()(ConvertArgs &args) const
  {
    double_dispatch(args.out.dim(), args.out.code(), ConvertImpl<KIND, SRC_TYPE>{}, args);
  }
};

template <VariantKind KIND>
static void convert_template(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context context,
                             Runtime *runtime)
{
  Deserializer ctx(task, regions);
  ConvertArgs args;
  deserialize(ctx, args);
  type_dispatch(args.in.code(), SourceTypeDispatch<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
