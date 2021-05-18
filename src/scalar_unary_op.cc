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

#include "scalar_unary_op.h"
#include "unary_op_util.h"
#include "core.h"
#include "dispatch.h"
#include "scalar.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <UnaryOpCode OP_CODE>
struct UnaryOpImpl {
  template <LegateTypeCode CODE, std::enable_if_t<UnaryOp<OP_CODE, CODE>::valid> * = nullptr>
  UntypedScalar operator()(UntypedScalar &in)
  {
    using OP  = UnaryOp<OP_CODE, CODE>;
    using ARG = legate_type_of<CODE>;

    OP func{};
    return UntypedScalar(func(in.value<ARG>()));
  }

  template <LegateTypeCode CODE, std::enable_if_t<!UnaryOp<OP_CODE, CODE>::valid> * = nullptr>
  UntypedScalar operator()(UntypedScalar &in)
  {
    assert(false);
    return UntypedScalar();
  }
};

struct UnaryOpDispatch {
  template <UnaryOpCode OP_CODE>
  UntypedScalar operator()(UntypedScalar &in)
  {
    return type_dispatch(in.code(), UnaryOpImpl<OP_CODE>{}, in);
  }
};

/*static*/ UntypedScalar ScalarUnaryOpTask::cpu_variant(const Task *task,
                                                        const std::vector<PhysicalRegion> &regions,
                                                        Context context,
                                                        Runtime *runtime)
{
  Deserializer ctx(task, regions);

  UnaryOpCode op_code;
  UntypedScalar in;

  deserialize(ctx, op_code);
  deserialize(ctx, in);

  return op_dispatch(op_code, UnaryOpDispatch{}, in);
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
