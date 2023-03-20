/* Copyright 20223 NVIDIA Corporation
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

#include "cunumeric/vectorize/eval_udf.h"
#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace legate;

struct EvalUdfCPU {
  template <LegateTypeCode CODE, int DIM>
  void operator()(EvalUdfArgs& args) const
  {
    // In the case of CPU, we pack arguments in a vector and pass them to the
    // function (through the function pointer geenrated by numba)
    using UDF = void(void**, size_t, size_t, uint32_t*, uint32_t*);
    auto udf  = reinterpret_cast<UDF*>(args.cpu_func_ptr);
    std::vector<void*> udf_args;
    size_t volume = 1;
    Pitches<DIM - 1> pitches;
    Rect<DIM> rect;
    size_t strides[DIM];
    if (args.inputs.size() > 0) {
      using VAL = legate_type_of<CODE>;
      rect      = args.inputs[0].shape<DIM>();
      volume    = pitches.flatten(rect);

      if (rect.empty()) return;
      for (size_t i = 0; i < args.inputs.size(); i++) {
        if (i < args.num_outputs) {
          auto out = args.outputs[i].write_accessor<VAL, DIM>(rect);
          udf_args.push_back(reinterpret_cast<void*>(out.ptr(rect, strides)));
        } else {
          auto out = args.inputs[i].read_accessor<VAL, DIM>(rect);
          udf_args.push_back(reinterpret_cast<void*>(const_cast<VAL*>(out.ptr(rect, strides))));
        }
      }
    }  // if
    for (auto s : args.scalars) udf_args.push_back(const_cast<void*>(s.ptr()));
    udf(udf_args.data(),
        volume,
        size_t(DIM),
        reinterpret_cast<uint32_t*>(const_cast<size_t*>(pitches.data())),
        reinterpret_cast<uint32_t*>(&strides[0]));
  }
};

/*static*/ void EvalUdfTask::cpu_variant(TaskContext& context)
{
  uint32_t num_outputs = context.scalars()[0].value<uint32_t>();
  uint32_t num_scalars = context.scalars()[1].value<uint32_t>();
  std::vector<Scalar> scalars;
  for (size_t i = 2; i < (2 + num_scalars); i++) scalars.push_back(context.scalars()[i]);

  EvalUdfArgs args{context.scalars()[2 + num_scalars].value<uint64_t>(),
                   context.inputs(),
                   context.outputs(),
                   scalars,
                   num_outputs,
                   context.get_current_processor()};
  size_t dim = 1;
  if (args.inputs.size() > 0) {
    dim = args.inputs[0].dim() == 0 ? 1 : args.inputs[0].dim();
    double_dispatch(dim, args.inputs[0].code(), EvalUdfCPU{}, args);
  } else {
    // FIXME
    double_dispatch(dim, args.inputs[0].code(), EvalUdfCPU{}, args);
  }
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { EvalUdfTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
