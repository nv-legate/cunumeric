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
struct DiagImplBody;

template <VariantKind KIND>
struct DiagImpl {
  template <LegateTypeCode CODE>
  void operator()(DiagArgs &args) const
  {
    using VAL = legate_type_of<CODE>;

    assert(args.shape.dim() == 2);

    const auto rect_mat = args.shape.to_rect<2>();
    // Solve for the start
    // y = x + k
    // x >= rect_mat.lo[0]
    const Point<2> start1(rect_mat.lo[0], rect_mat.lo[0] + args.k);
    // y >= rect_mat.lo[1]
    const Point<2> start2(rect_mat.lo[1] - args.k, rect_mat.lo[1]);
    // See if our rect_mat intersects with the diagonal
    if (!rect_mat.contains(start1) && !rect_mat.contains(start2)) return;

    // Pick whichever one fits in our rect
    const Point<2> start = rect_mat.contains(start1) ? start1 : start2;
    // Now do the same thing for the end
    // x <= rect_mat.hi[0]
    const Point<2> stop1(rect_mat.hi[0], rect_mat.hi[0] + args.k);
    // y <= rect_mat.hi[1]
    const Point<2> stop2(rect_mat.hi[1] - args.k, rect_mat.hi[1]);
    assert(rect_mat.contains(stop1) || rect_mat.contains(stop2));
    const Point<2> stop = rect_mat.contains(stop1) ? stop1 : stop2;
    // Walk the path from the stop to the start
    const coord_t distance = (stop[0] - start[0]) + 1;
    // Should be the same along both dimensions
    assert(distance == ((stop[1] - start[1]) + 1));

    if (args.extract) {
      assert(args.out.dim() == 1);
      assert(args.in.dim() == 2);

      auto rect1d = args.out.shape<1>();
      if (rect1d.empty()) return;

      auto in = args.in.read_accessor<VAL, 2>(rect_mat);

      auto start_out = args.k > 0 ? start[0] : start[1];
      if (args.needs_reduction) {
        auto out = args.out.reduce_accessor<SumReduction<VAL>, true, 1>(rect1d);
        DiagImplBody<KIND, CODE>()(out, in, distance, start_out, start);
      } else {
        auto out = args.out.write_accessor<VAL, 1>(rect1d);
        DiagImplBody<KIND, CODE>()(out, in, distance, start_out, start);
      }
    } else {
      assert(args.out.dim() == 2);
      assert(args.in.dim() == 1);

      auto rect1d = args.in.shape<1>();
      if (rect1d.empty()) return;

      auto out = args.out.write_accessor<VAL, 2>(rect_mat);
      auto in  = args.in.read_accessor<VAL, 1>(rect1d);

      auto start_in = args.k > 0 ? start[0] : start[1];
      if (args.k > 0)
        DiagImplBody<KIND, CODE>()(out, in, distance, start, start_in);
      else
        DiagImplBody<KIND, CODE>()(out, in, distance, start, start_in);
    }
  }
};

template <VariantKind KIND>
static void diag_template(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context context,
                          Runtime *runtime)
{
  Deserializer ctx(task, regions);
  DiagArgs args;
  deserialize(ctx, args);
  type_dispatch(args.in.code(), DiagImpl<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
