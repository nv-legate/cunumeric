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

namespace legate {
namespace numpy {

using namespace Legion;

template <VariantKind KIND, LegateTypeCode CODE>
struct BincountImplBody;

template <VariantKind KIND>
struct BincountImpl {
  template <LegateTypeCode CODE, std::enable_if_t<is_integral<CODE>::value> * = nullptr>
  void operator()(BincountArgs &args) const
  {
    using VAL = legate_type_of<CODE>;

    auto rect     = args.shape.to_rect<1>();
    auto lhs_rect = args.lhs.shape<1>();
    if (rect.empty()) return;

    auto rhs = args.rhs.read_accessor<VAL, 1>();
    if (args.has_weights) {
      auto weights = args.weights.read_accessor<double, 1>();
      if (args.needs_reduction) {
        auto lhs = args.lhs.reduce_accessor<SumReduction<double>, KIND != VariantKind::GPU, 1>();
        BincountImplBody<KIND, CODE>()(lhs, rhs, weights, rect, lhs_rect);
      } else {
        auto lhs = args.lhs.read_write_accessor<double, 1>();
        BincountImplBody<KIND, CODE>()(lhs, rhs, weights, rect, lhs_rect);
      }
    } else {
      if (args.needs_reduction) {
        auto lhs = args.lhs.reduce_accessor<SumReduction<int64_t>, KIND != VariantKind::GPU, 1>();
        BincountImplBody<KIND, CODE>()(lhs, rhs, rect, lhs_rect);
      } else {
        auto lhs = args.lhs.read_write_accessor<int64_t, 1>();
        BincountImplBody<KIND, CODE>()(lhs, rhs, rect, lhs_rect);
      }
    }
  }

  template <LegateTypeCode CODE, std::enable_if_t<!is_integral<CODE>::value> * = nullptr>
  void operator()(BincountArgs &args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void bincount_template(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context context,
                              Runtime *runtime)
{
  Deserializer ctx(task, regions);
  BincountArgs args;
  deserialize(ctx, args);
  type_dispatch(args.rhs.code(), BincountImpl<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
