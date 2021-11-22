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

#include "cunumeric/transform/flip.h"
#include "cunumeric/transform/flip_template.inl"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE, int32_t DIM>
struct FlipImplBody<VariantKind::CPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, DIM> in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  legate::Span<const int32_t> axes) const

  {
    for (PointInRectIterator<DIM> itr(rect); itr.valid(); ++itr) {
      auto q = *itr;
      for (uint32_t idx = 0; idx < axes.size(); ++idx)
        q[axes[idx]] = rect.hi[axes[idx]] - q[axes[idx]];
      out[*itr] = in[q];
    }
  }
};

/*static*/ void FlipTask::cpu_variant(TaskContext& context)
{
  flip_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { FlipTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
