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

    auto shape = args.matrix.shape<2>();

    // Solve for the start
    // y = x
    // x >= shape.lo[0]
    const Point<2> start1(shape.lo[0], shape.lo[0]);
    // y >= shape.lo[1]
    const Point<2> start2(shape.lo[1], shape.lo[1]);
    // See if our shape intersects with the diagonal
    if (!shape.contains(start1) && !shape.contains(start2)) return;

    // Pick whichever one fits in our rect
    const Point<2> start = shape.contains(start1) ? start1 : start2;
    // Now do the same thing for the end
    // x <= shape.hi[0]
    const Point<2> stop1(shape.hi[0], shape.hi[0]);
    // y <= shape.hi[1]
    const Point<2> stop2(shape.hi[1], shape.hi[1]);
    assert(shape.contains(stop1) || shape.contains(stop2));
    const Point<2> stop = shape.contains(stop1) ? stop1 : stop2;
    // Walk the path from the stop to the start
    const coord_t distance = (stop[0] - start[0]) + 1;
    // Should be the same along both dimensions
    assert(distance == ((stop[1] - start[1]) + 1));

    if (args.extract) {
      auto in  = args.matrix.read_accessor<VAL, 2>(shape);
      auto out = args.diag.reduce_accessor<SumReduction<VAL>, true, 2>(shape);
      DiagImplBody<KIND, CODE>()(out, in, start, distance);
    } else {
      auto in  = args.diag.read_accessor<VAL, 2>(shape);
      auto out = args.matrix.write_accessor<VAL, 2>(shape);
      DiagImplBody<KIND, CODE>()(out, in, start, distance);
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
  type_dispatch(args.matrix.code(), DiagImpl<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
