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

template <VariantKind KIND, LegateTypeCode CODE, int32_t DIM>
struct TransposeImplBody;

template <VariantKind KIND>
struct TransposeImpl {
  template <LegateTypeCode CODE, int32_t DIM, std::enable_if_t<DIM == 2> * = nullptr>
  void operator()(TransposeArgs &args) const
  {
    using VAL = legate_type_of<CODE>;

    const auto out_rect = args.out.shape<2>();
    if (out_rect.empty()) return;
    const Rect<2> in_rect(Point<2>(out_rect.lo[1], out_rect.lo[0]),
                          Point<2>(out_rect.hi[1], out_rect.hi[0]));

    auto out = args.out.write_accessor<VAL, DIM>();
    auto in  = args.in.read_accessor<VAL, DIM>();

    TransposeImplBody<KIND, CODE, DIM>{}(out_rect, in_rect, out, in);
  }

  template <LegateTypeCode CODE, int32_t DIM, std::enable_if_t<DIM != 2> * = nullptr>
  void operator()(TransposeArgs &args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void transpose_template(TaskContext &context)
{
  TransposeArgs args{context.outputs()[0], context.inputs()[0]};
  double_dispatch(args.out.dim(), args.in.code(), TransposeImpl<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
