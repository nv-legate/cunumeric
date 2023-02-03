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
#include "cunumeric/vectorize/eval_udf.h"
#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, LegateTypeCode CODE, int DIM>
struct EvalUdfImplBody;

template <VariantKind KIND>
struct EvalUdfImpl {
  template <LegateTypeCode CODE, int DIM>
  void operator()(EvalUdfArgs& args) const
  {
    using UDF = void(void**, size_t);
    auto udf  = reinterpret_cast<UDF*>(args.func_ptr);
    std::vector<void*> udf_args;
    using VAL = legate_type_of<CODE>;
    auto rect = args.args[0].shape<DIM>();

    if (rect.empty()) return;
    EvalUdfImplBody<KIND, CODE, DIM>();
    for (size_t i = 0; i < args.args.size(); i++) {
      auto out = args.args[i].write_accessor<VAL, DIM>(rect);
      udf_args.push_back(reinterpret_cast<void*>(out.ptr(rect)));
    }

    udf(udf_args.data(), rect.volume());
  }
};

template <VariantKind KIND>
static void eval_udf_template(TaskContext& context)
{
  is_gpus = context.scalars()[0].value<bool>();
  if (is_gpus)
    std::cout << "IRINA DEBUG size of the scalars = " << context.scalars().size() << std::endl;
  EvalUdfArgs args{0, context.scalars()[1].value<char*>(), context.outputs()};
  else EvalUdfArgs args{context.scalars()[1].value<uint64_t>(),'', context.outputs()};
  size_t dim = args.args[0].dim() == 0 ? 1 : args.args[0].dim();
  double_dispatch(dim, args.args[0].code(), EvalUdfImpl<KIND>{}, args);
}

}  // namespace cunumeric
