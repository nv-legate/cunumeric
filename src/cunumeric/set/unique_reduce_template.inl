/* Copyright 2022 NVIDIA Corporation
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

#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, LegateTypeCode CODE>
struct UniqueReduceImplBody;

template <VariantKind KIND>
struct UniqueReduceImpl {
  template <LegateTypeCode CODE>
  void operator()(Array& output, std::vector<Array>& input_arrs)
  {
    using VAL = legate_type_of<CODE>;

    std::vector<std::pair<AccessorRO<VAL, 1>, Rect<1>>> inputs;

    for (auto& input_arr : input_arrs) {
      auto shape = input_arr.shape<1>();
      auto acc   = input_arr.read_accessor<VAL, 1>(shape);
      inputs.push_back(std::make_pair(acc, shape));
    }

    size_t size;
    Buffer<VAL> result;
    std::tie(result, size) = UniqueReduceImplBody<KIND, CODE>()(inputs);

    output.return_data(result, size);
  }
};

template <VariantKind KIND>
static void unique_reduce_template(TaskContext& context)
{
  auto& inputs = context.inputs();
  auto& output = context.outputs()[0];
  type_dispatch(output.code(), UniqueReduceImpl<KIND>{}, output, inputs);
}

}  // namespace cunumeric
