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

#include "unary/scalar_unary_op.h"
#include "unary/unary_op_util.h"
#include "scalar.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <UnaryOpCode OP_CODE>
struct UnaryOpImpl {
  template <LegateTypeCode CODE, std::enable_if_t<UnaryOp<OP_CODE, CODE>::valid>* = nullptr>
  UntypedScalar operator()(const UntypedScalar& in, const std::vector<UntypedScalar>& args) const
  {
    using OP  = UnaryOp<OP_CODE, CODE>;
    using ARG = legate_type_of<CODE>;

    OP func{args};

    return UntypedScalar(func(in.value<ARG>()));
  }

  template <LegateTypeCode CODE, std::enable_if_t<!UnaryOp<OP_CODE, CODE>::valid>* = nullptr>
  UntypedScalar operator()(const UntypedScalar& in, const std::vector<UntypedScalar>& args) const
  {
    assert(false);
    return UntypedScalar();
  }
};

struct UnaryOpDispatch {
  template <UnaryOpCode OP_CODE>
  UntypedScalar operator()(const UntypedScalar& in, const std::vector<UntypedScalar>& args) const
  {
    return type_dispatch(in.code(), UnaryOpImpl<OP_CODE>{}, in, args);
  }
};

/*static*/ UntypedScalar ScalarUnaryOpTask::cpu_variant(TaskContext& context)
{
  auto op_code = context.scalars()[0].value<UnaryOpCode>();

  auto& inputs = context.inputs();

  auto& in = inputs[0];

  std::vector<UntypedScalar> args;
  for (auto idx = 1; idx < inputs.size(); ++idx)
    args.push_back(inputs[idx].scalar<UntypedScalar>());

  return op_dispatch(op_code, UnaryOpDispatch{}, in.scalar<UntypedScalar>(), args);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  ScalarUnaryOpTask::register_variants_with_return<UntypedScalar, UntypedScalar>();
}
}  // namespace

}  // namespace numpy
}  // namespace legate
