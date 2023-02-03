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

namespace cunumeric {

using namespace Legion;
using namespace legate;

struct EvalUdfCPU {
  template <LegateTypeCode CODE, int DIM>
  void operator()(EvalUdfArgs& args) const
  {
    std::cout << "IRINA DEBUG in CPU task 2" << std::endl;
    using UDF = void(void**, size_t);
    auto udf  = reinterpret_cast<UDF*>(args.cpu_func_ptr);
    std::vector<void*> udf_args;
    using VAL = legate_type_of<CODE>;
    auto rect = args.args[0].shape<DIM>();

    if (rect.empty()) return;
    for (size_t i = 0; i < args.args.size(); i++) {
      auto out = args.args[i].write_accessor<VAL, DIM>(rect);
      udf_args.push_back(reinterpret_cast<void*>(out.ptr(rect)));
    }

    udf(udf_args.data(), rect.volume());
  }
};

/*static*/ void EvalUdfTask::cpu_variant(TaskContext& context)
{
  std::cout << "IRINA DEBUG in CPU task" << std::endl;
  EvalUdfArgs args{context.scalars()[0].value<uint64_t>(), context.outputs()};
  size_t dim = args.args[0].dim() == 0 ? 1 : args.args[0].dim();
  double_dispatch(dim, args.args[0].code(), EvalUdfCPU{}, args);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { EvalUdfTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
