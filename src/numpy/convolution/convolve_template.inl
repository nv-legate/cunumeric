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
    auto subrect     = args.out.shape<DIM>();
    auto filter_rect = args.filter.shape<DIM>();

    if (subrect.empty()) return;

    auto out    = args.out.write_accessor<VAL, DIM>(subrect);
    auto filter = args.filter.read_accessor<VAL, DIM>(filter_rect);
    auto input  = args.inputs[0].read_accessor<VAL, DIM>(subrect);

    Rect<DIM> root_rect(args.root_domain);
    ConvolveImplBody<KIND, CODE, DIM>()(out, filter, input, root_rect, subrect, filter_rect);
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
  ConvolveArgs args;

  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();

  args.out    = std::move(outputs[0]);
  args.filter = std::move(inputs[0]);
  for (uint32_t idx = 1; idx < inputs.size(); ++idx) args.inputs.push_back(std::move(inputs[idx]));

  auto shape           = context.scalars()[0].value<DomainPoint>();
  args.root_domain.dim = shape.dim;
  for (int32_t dim = 0; dim < shape.dim; ++dim) {
    args.root_domain.rect_data[dim]             = 0;
    args.root_domain.rect_data[dim + shape.dim] = shape[dim] - 1;
  }

  double_dispatch(args.out.dim(), args.out.code(), ConvolveImpl<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
