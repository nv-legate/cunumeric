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

#include "cunumeric/nullary/eye.h"
#include "cunumeric/nullary/eye_template.inl"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <typename VAL>
struct EyeImplBody<VariantKind::OMP, VAL> {
  void operator()(const AccessorWO<VAL, 2>& out,
                  const Point<2>& start,
                  const coord_t distance) const
  {
#pragma omp parallel for
    for (coord_t idx = 0; idx < distance; idx++) out[start[0] + idx][start[1] + idx] = VAL{1};
  }
};

/*static*/ void EyeTask::omp_variant(TaskContext& context)
{
  eye_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
