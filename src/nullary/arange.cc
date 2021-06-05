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

#include "nullary/arange.h"
#include "nullary/arange_template.inl"

namespace legate {
namespace numpy {

using namespace Legion;

template <typename VAL>
struct ArangeImplBody<VariantKind::CPU, VAL> {
  void operator()(const AccessorWO<VAL, 1> &out,
                  const Rect<1> &rect,
                  const VAL start,
                  const VAL step) const
  {
    for (coord_t idx = rect.lo[0]; idx <= rect.hi[0]; ++idx)
      out[idx] = static_cast<VAL>(idx) * step + start;
  }
};

void deserialize(Deserializer &ctx, ArangeArgs &args)
{
  deserialize(ctx, args.shape);
  deserialize(ctx, args.out);
  deserialize(ctx, args.start);
  deserialize(ctx, args.stop);
  deserialize(ctx, args.step);
}

/*static*/ void ArangeTask::cpu_variant(const Task *task,
                                        const std::vector<PhysicalRegion> &regions,
                                        Context context,
                                        Runtime *runtime)
{
  arange_template<VariantKind::CPU>(task, regions, context, runtime);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { ArangeTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
