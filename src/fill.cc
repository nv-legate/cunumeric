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

#include "fill.h"
#include "core.h"
#include "dispatch.h"
#include "point_task.h"
#include "scalar.h"

namespace legate {
namespace numpy {

using namespace Legion;

struct FillImpl {
  template <LegateTypeCode CODE, int DIM>
  void operator()(Shape &shape, RegionField &out_rf, UntypedScalar &fill_value_scalar)
  {
    using VAL = legate_type_of<CODE>;

    auto rect = shape.to_rect<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out        = out_rf.write_accessor<VAL, DIM>();
    auto fill_value = fill_value_scalar.value<VAL>();

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    if (dense) {
      auto outptr = out.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) outptr[idx] = fill_value;
    } else
      for (size_t idx = 0; idx < volume; ++idx) {
        const auto point = pitches.unflatten(idx, rect.lo);
        out[point]       = fill_value;
      }
  }
};

/*static*/ void FillTask::cpu_variant(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context context,
                                      Runtime *runtime)
{
  Deserializer ctx(task, regions);

  Shape shape;
  RegionField out;
  UntypedScalar fill_value;

  deserialize(ctx, shape);
  deserialize(ctx, out);
  deserialize(ctx, fill_value);

  double_dispatch(out.dim(), out.code(), FillImpl{}, shape, out, fill_value);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { FillTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
