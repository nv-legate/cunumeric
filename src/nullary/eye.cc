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

#include "nullary/eye.h"
#include "nullary/eye_template.inl"

namespace legate {
namespace numpy {

using namespace Legion;

template <typename VAL>
struct EyeImplBody<VariantKind::CPU, VAL> {
  void operator()(const AccessorWO<VAL, 2> &out,
                  const Point<2> &start,
                  const coord_t distance) const
  {
    for (coord_t idx = 0; idx < distance; idx++) out[start[0] + idx][start[1] + idx] = VAL{1};
  }
};

void deserialize(Deserializer &ctx, EyeArgs &args)
{
  deserialize(ctx, args.out);
  Scalar k;
  deserialize(ctx, k);
  args.k = k.value<int32_t>();
}

/*static*/ void EyeTask::cpu_variant(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context context,
                                     Runtime *runtime)
{
  eye_template<VariantKind::CPU>(task, regions, context, runtime);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { EyeTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
