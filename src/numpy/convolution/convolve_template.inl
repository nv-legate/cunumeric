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

#include "numpy/pitches.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <VariantKind KIND, LegateTypeCode CODE, int DIM>
struct ConvolveImplBody;

template <VariantKind KIND>
struct ConvolveImpl {
  template <LegateTypeCode CODE, int DIM, std::enable_if_t<(DIM <= 3)>* = nullptr>
  void operator()(ConvolveArgs& args) const
  {
    using VAL        = legate_type_of<CODE>;
    auto out_rect    = args.out.shape<DIM>();
    auto filter_rect = args.in2.shape<DIM>();

    if (out_rect.empty()) return;

    auto out = args.out.write_accessor<VAL, DIM>(out_rect);
    auto in1 = args.in1.read_accessor<VAL, DIM>(out_rect);
    auto in2 = args.in2.read_accessor<VAL, DIM>(filter_rect);

    ConvolveImplBody<KIND, CODE, DIM>()(out, in1, in2, out_rect, filter_rect);
  }

  template <LegateTypeCode CODE, int DIM, std::enable_if_t<!(DIM <= 3)>* = nullptr>
  void operator()(ConvolveArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void convolve_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();

  ConvolveArgs args{inputs[0], inputs[1], outputs[0]};
  double_dispatch(args.out.dim(), args.out.code(), ConvolveImpl<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
