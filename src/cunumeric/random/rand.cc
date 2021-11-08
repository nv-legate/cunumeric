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

#include "cunumeric/random/rand.h"
#include "cunumeric/random/rand_template.inl"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <typename RNG, typename VAL, int32_t DIM>
struct RandImplBody<VariantKind::CPU, RNG, VAL, DIM> {
  void operator()(AccessorWO<VAL, DIM> out,
                  const RNG& rng,
                  const Point<DIM>& strides,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect) const
  {
    size_t volume = rect.volume();
    for (size_t idx = 0; idx < volume; ++idx) {
      const auto point = pitches.unflatten(idx, rect.lo);
      size_t offset    = 0;
      for (size_t dim = 0; dim < DIM; ++dim) offset += point[dim] * strides[dim];
      out[point] = rng(HI_BITS(offset), LO_BITS(offset));
    }
  }
};

/*static*/ void RandTask::cpu_variant(TaskContext& context)
{
  rand_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { RandTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
