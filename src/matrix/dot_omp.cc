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

#include "matrix/dot.h"
#include "matrix/dot_template.inl"

#include <omp.h>
#include <alloca.h>

namespace legate {
namespace numpy {

using namespace Legion;

template <LegateTypeCode CODE>
struct DotImplBody<VariantKind::OMP, CODE> {
  using VAL = legate_type_of<CODE>;

  void operator()(VAL& result,
                  const AccessorRO<VAL, 1>& rhs1,
                  const AccessorRO<VAL, 1>& rhs2,
                  const Rect<1>& rect,
                  bool dense)
  {
    const auto volume      = rect.volume();
    const auto max_threads = omp_get_max_threads();
    auto locals            = static_cast<VAL*>(alloca(max_threads * sizeof(VAL)));
    for (auto idx = 0; idx < max_threads; ++idx) locals[idx] = SumReduction<VAL>::identity;

    if (dense) {
      auto rhs1ptr = rhs1.ptr(rect);
      auto rhs2ptr = rhs2.ptr(rect);
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (size_t idx = 0; idx < volume; ++idx) {
          const VAL prod = rhs1ptr[idx] * rhs2ptr[idx];
          SumReduction<VAL>::template fold<true>(locals[tid], prod);
        }
      }
    } else {
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (coord_t idx = rect.lo[0]; idx <= rect.hi[0]; ++idx) {
          const VAL prod = rhs1[idx] * rhs2[idx];
          SumReduction<VAL>::template fold<true>(locals[tid], prod);
        }
      }
    }

    for (auto idx = 0; idx < max_threads; ++idx)
      SumReduction<VAL>::template fold<true>(result, locals[idx]);
  }
};

/*static*/ UntypedScalar DotTask::omp_variant(TaskContext& context)
{
  return dot_template<VariantKind::OMP>(context);
}

}  // namespace numpy
}  // namespace legate
