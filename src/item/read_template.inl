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

template <VariantKind KIND, typename VAL>
struct ReadImplBody;

template <VariantKind KIND>
struct ReadImpl {
  template <LegateTypeCode CODE>
  UntypedScalar operator()(const Array& in_arr) const
  {
    using VAL = legate_type_of<CODE>;
    auto in   = in_arr.read_accessor<VAL, 1>();
    return ReadImplBody<KIND, VAL>()(in);
  }
};

template <VariantKind KIND>
static UntypedScalar read_template(TaskContext& context)
{
  auto& in = context.inputs()[0];
  return type_dispatch(in.code(), ReadImpl<KIND>{}, in);
}

}  // namespace numpy
}  // namespace legate
