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

#include "core.h"
#include "deserializer.h"
#include "dispatch.h"
#include "scalar.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <VariantKind KIND, typename VAL, int DIM>
struct WriteImplBody;

template <VariantKind KIND>
struct WriteImpl {
  template <LegateTypeCode CODE, int DIM>
  void operator()(Array &out_arr, UntypedPoint &key, const UntypedScalar &scalar) const
  {
    using VAL = legate_type_of<CODE>;
    auto out  = out_arr.write_accessor<VAL, DIM>();
    WriteImplBody<KIND, VAL, DIM>()(out, key.to_point<DIM>(), scalar.value<VAL>());
  }
};

template <VariantKind KIND>
static void write_template(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context context,
                           Runtime *runtime)
{
  Deserializer ctx(task, regions);
  UntypedPoint key;
  Array out;
  UntypedScalar scalar;
  deserialize(ctx, key);
  deserialize(ctx, out);
  deserialize(ctx, scalar);
  double_dispatch(out.dim(), out.code(), WriteImpl<KIND>{}, out, key, scalar);
}

}  // namespace numpy
}  // namespace legate
