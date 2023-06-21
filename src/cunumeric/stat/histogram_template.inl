/* Copyright 2023 NVIDIA Corporation
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

#pragma once

// Useful for IDEs
#include "cunumeric/stat/histogram.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct HistogramImplBody;

template <VariantKind KIND>
struct HistogramImpl {
  template <Type::Code CODE>
  void operator()(HistogramArgs& args) const
  {
    using VAL = legate_type_of<CODE>;

    auto result  = args.result.shape<1>();
    auto src     = args.src.shape<1>();
    auto bins    = args.bins.shape<1>();
    auto weights = args.weights.shape<1>();

    if (src.empty()) return;

    // TODO...
    HistogramImplBody<KIND, CODE>()(src, bins, weights, result);
  }
};

template <VariantKind KIND>
static void histogram_template(TaskContext& context)
{
  auto& inputs     = context.inputs();
  auto& reductions = context.reductions();
  HistogramArgs args{reductions[0], inputs[0], inputs[1], inputs[2]};
  type_dispatch(args.rhs.code(), HistogramImpl<KIND>{}, args);
}

}  // namespace cunumeric