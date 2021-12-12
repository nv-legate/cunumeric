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

#include "cunumeric/matrix/trilu.h"
#include "cunumeric/matrix/trilu_template.inl"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE, int32_t DIM, bool LOWER>
struct TriluImplBody<VariantKind::OMP, CODE, DIM, LOWER> {
  using VAL = legate_type_of<CODE>;

  void operator()(const AccessorWO<VAL, DIM>& out,
                  const AccessorRO<VAL, DIM>& in,
                  const Pitches<DIM - 1>& pitches,
                  const Point<DIM>& lo,
                  size_t volume,
                  int32_t k) const
  {
    if (LOWER)
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, lo);
        if (p[DIM - 2] + k >= p[DIM - 1])
          out[p] = in[p];
        else
          out[p] = 0;
      }
    else
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, lo);
        if (p[DIM - 2] + k <= p[DIM - 1])
          out[p] = in[p];
        else
          out[p] = 0;
      }
  }
};

/*static*/ void TriluTask::omp_variant(TaskContext& context)
{
  trilu_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
