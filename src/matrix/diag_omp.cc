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
struct DiagImplBody<VariantKind::OMP, CODE> {
  using VAL = legate_type_of<CODE>;

  void operator()(const AccessorWO<VAL, 2>& out,
                  const AccessorRO<VAL, 2>& in,
                  const Point<2>& start,
                  size_t distance) const
  {
#pragma omp parallel for schedule(static)
    for (coord_t idx = 0; idx < distance; ++idx) {
      Point<2> p(start[0] + idx, start[1] + idx);
      out[p] = in[p];
    }
  }

  void operator()(const AccessorRD<SumReduction<VAL>, true, 2>& out,
                  const AccessorRO<VAL, 2>& in,
                  const Point<2>& start,
                  size_t distance) const
  {
#pragma omp parallel for schedule(static)
    for (coord_t idx = 0; idx < distance; ++idx) {
      Point<2> p(start[0] + idx, start[1] + idx);
      auto v = in[p];
      out.reduce(p, v);
    }
  }
};

/*static*/ void DiagTask::omp_variant(TaskContext& context)
{
  diag_template<VariantKind::OMP>(context);
}

}  // namespace numpy
}  // namespace legate
