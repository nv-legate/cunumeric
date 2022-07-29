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

#include "cunumeric/cunumeric.h"
#include "cunumeric/unary/scalar_unary_red.h"
#include "cunumeric/unary/scalar_unary_red_template.inl"
#include "cunumeric/omp_help.h"

#include <omp.h>

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <>
struct ScalarUnaryRedImplBody<VariantKind::OMP> {
  template <class AccessorRD, class Kernel, class LHS>
  void operator()(AccessorRD& out, size_t volume, const LHS& identity, Kernel&& kernel)
  {
    const auto max_threads = omp_get_max_threads();
    ThreadLocalStorage<LHS> locals(max_threads);
    for (auto idx = 0; idx < max_threads; ++idx) locals[idx] = identity;
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) { kernel(locals[tid], idx); }
    }
    for (auto idx = 0; idx < max_threads; ++idx) out.reduce(0, locals[idx]);
  }
};

/*static*/ void ScalarUnaryRedTask::omp_variant(TaskContext& context)
{
  scalar_unary_red_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
