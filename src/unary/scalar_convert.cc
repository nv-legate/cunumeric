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

#include "unary/scalar_convert.h"
#include "unary/convert_util.h"
#include "core.h"
#include "dispatch.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <LegateTypeCode SRC_TYPE>
struct ConvertImpl {
  template <LegateTypeCode DST_TYPE, std::enable_if_t<SRC_TYPE != DST_TYPE> * = nullptr>
  UntypedScalar operator()(const UntypedScalar &in_scalar) const
  {
    using OP  = ConvertOp<DST_TYPE, SRC_TYPE>;
    using SRC = legate_type_of<SRC_TYPE>;
    using DST = legate_type_of<DST_TYPE>;

    OP func{};
    auto in = in_scalar.value<SRC>();
    return UntypedScalar(func(in));
  }

  template <LegateTypeCode DST_TYPE, std::enable_if_t<SRC_TYPE == DST_TYPE> * = nullptr>
  UntypedScalar operator()(const UntypedScalar &in_scalar) const
  {
    assert(false);
    return UntypedScalar();
  }
};

struct SourceTypeDispatch {
  template <LegateTypeCode SRC_TYPE>
  UntypedScalar operator()(LegateTypeCode dtype, const UntypedScalar &in) const
  {
    return type_dispatch(dtype, ConvertImpl<SRC_TYPE>{}, in);
  }
};

/*static*/ UntypedScalar ScalarConvertTask::cpu_variant(const Task *task,
                                                        const std::vector<PhysicalRegion> &regions,
                                                        Context context,
                                                        Runtime *runtime)
{
  Deserializer ctx(task, regions);

  LegateTypeCode dtype;
  Array in;
  deserialize(ctx, dtype);
  deserialize(ctx, in);

  return type_dispatch(in.code(), SourceTypeDispatch{}, dtype, in.scalar<UntypedScalar>());
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  ScalarConvertTask::register_variants_with_return<UntypedScalar, UntypedScalar>();
}
}  // namespace

}  // namespace numpy
}  // namespace legate
