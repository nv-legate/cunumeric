/* Copyright 2021-2022 NVIDIA Corporation
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

#include "cunumeric/matrix/diag.h"
#include "cunumeric/matrix/diag_template.inl"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE, int DIM>
struct DiagImplBody<VariantKind::CPU, CODE, DIM, true> {
  using VAL = legate_type_of<CODE>;

  void operator()(const AccessorRD<SumReduction<VAL>, true, DIM>& out,
                  const AccessorRO<VAL, DIM>& in,
                  const coord_t& start,
                  const Pitches<DIM - 1>& m_pitches,
                  const Rect<DIM>& m_shape,
                  const size_t naxes,
                  const coord_t distance) const
  {
    size_t skip_size = 1;

    for (int i = 0; i < naxes; i++) {
      auto diff = 1 + m_shape.hi[DIM - i - 1] - m_shape.lo[DIM - i - 1];
      if (diff != 0) skip_size *= diff;
    }
    const size_t volume = m_shape.volume();
    for (size_t idx = 0; idx < volume; idx += skip_size) {
      Point<DIM> p = m_pitches.unflatten(idx, m_shape.lo);
      for (coord_t d = 0; d < distance; ++d) {
        for (size_t i = DIM - naxes; i < DIM; i++) { p[i] = start + d; }
        auto v = in[p];
        out.reduce(p, v);
      }
    }
  }
};

// not extract (create a new 2D matrix with diagonal from vector)
template <LegateTypeCode CODE>
struct DiagImplBody<VariantKind::CPU, CODE, 2, false> {
  using VAL = legate_type_of<CODE>;

  void operator()(const AccessorRO<VAL, 2>& in,
                  const AccessorRW<VAL, 2>& out,
                  const Point<2>& start,
                  const coord_t distance)
  {
    for (coord_t idx = 0; idx < distance; idx++) {
      Point<2> p(start[0] + idx, start[1] + idx);
      out[p] = in[p];
    }
  }
};

/*static*/ void DiagTask::cpu_variant(TaskContext& context)
{
  diag_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { DiagTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
