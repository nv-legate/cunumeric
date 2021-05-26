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

#include "matrix/diag.h"
#include "matrix/diag_template.inl"

namespace legate {
namespace numpy {

using namespace Legion;

template <LegateTypeCode CODE>
struct DiagImplBody<VariantKind::CPU, CODE> {
  using VAL = legate_type_of<CODE>;

  void operator()(const AccessorWO<VAL, 2> &out,
                  const AccessorRO<VAL, 1> &in,
                  coord_t distance,
                  const Point<2> &start_out,
                  coord_t start_in) const
  {
    for (coord_t idx = 0; idx < distance; ++idx)
      out[start_out[0] + idx][start_out[1] + idx] = in[start_in + idx];
  }

  void operator()(const AccessorWO<VAL, 1> &out,
                  const AccessorRO<VAL, 2> &in,
                  coord_t distance,
                  coord_t start_out,
                  const Point<2> &start_in) const
  {
    for (coord_t idx = 0; idx < distance; ++idx)
      out[start_out + idx] = in[start_in[0] + idx][start_in[1] + idx];
  }

  void operator()(const AccessorRD<SumReduction<VAL>, true, 1> &out,
                  const AccessorRO<VAL, 2> &in,
                  coord_t distance,
                  coord_t start_out,
                  const Point<2> &start_in) const
  {
    for (coord_t idx = 0; idx < distance; ++idx)
      out.reduce(start_out + idx, in[start_in[0] + idx][start_in[1] + idx]);
  }
};

void deserialize(Deserializer &ctx, DiagArgs &args)
{
  deserialize(ctx, args.extract);
  deserialize(ctx, args.needs_reduction);
  deserialize(ctx, args.k);
  deserialize(ctx, args.shape);
  deserialize(ctx, args.out);
  deserialize(ctx, args.in);
}

/*static*/ void DiagTask::cpu_variant(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context context,
                                      Runtime *runtime)
{
  diag_template<VariantKind::CPU>(task, regions, context, runtime);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { DiagTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
