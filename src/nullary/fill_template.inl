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

#include "arg.h"
#include "core.h"
#include "dispatch.h"
#include "point_task.h"
#include "scalar.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <VariantKind KIND, typename VAL, int DIM>
struct FillImplBody;

template <VariantKind KIND>
struct FillImpl {
  template <typename VAL, int DIM>
  void fill(FillArgs &args) const
  {
    auto rect = args.shape.to_rect<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out        = args.out.write_accessor<VAL, DIM>();
    auto fill_value = args.fill_value.value<VAL>();

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif
    FillImplBody<KIND, VAL, DIM>{}(out, fill_value, pitches, rect, dense);
  }

  template <LegateTypeCode CODE, int DIM>
  void operator()(FillArgs &args) const
  {
    if (args.fill_value.is_argval()) {
      using VAL = Argval<legate_type_of<CODE>>;
      fill<VAL, DIM>(args);
    } else {
      using VAL = legate_type_of<CODE>;
      fill<VAL, DIM>(args);
    }
  }
};

template <VariantKind KIND>
static void fill_template(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context context,
                          Runtime *runtime)
{
  Deserializer ctx(task, regions);
  FillArgs args;
  deserialize(ctx, args);
  double_dispatch(args.out.dim(), args.out.code(), FillImpl<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
