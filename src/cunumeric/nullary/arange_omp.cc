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

#include "cunumeric/nullary/arange.h"
#include "cunumeric/nullary/arange_template.inl"

namespace cunumeric {

using namespace legate;

template <typename VAL>
struct ArangeImplBody<VariantKind::OMP, VAL> {
  void operator()(const AccessorWO<VAL, 1>& out,
                  const Rect<1>& rect,
                  const VAL start,
                  const VAL step) const
  {
#pragma omp parallel for
    for (coord_t idx = rect.lo[0]; idx <= rect.hi[0]; ++idx)
      out[idx] = static_cast<VAL>(idx) * step + start;
  }
};

/*static*/ void ArangeTask::omp_variant(TaskContext& context)
{
  arange_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
