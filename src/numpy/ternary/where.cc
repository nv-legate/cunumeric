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

#include "numpy/ternary/where.h"
#include "numpy/ternary/where_template.inl"

namespace legate {
namespace numpy {

using namespace Legion;

template <LegateTypeCode CODE, int DIM>
struct WhereImplBody<VariantKind::CPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<bool, DIM> mask,
                  AccessorRO<VAL, DIM> in1,
                  AccessorRO<VAL, DIM> in2,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    if (dense) {
      size_t volume = rect.volume();
      auto outptr   = out.ptr(rect);
      auto maskptr  = mask.ptr(rect);
      auto in1ptr   = in1.ptr(rect);
      auto in2ptr   = in2.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx)
        outptr[idx] = maskptr[idx] ? in1ptr[idx] : in2ptr[idx];
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto point = pitches.unflatten(idx, rect.lo);
        out[point] = mask[point] ? in1[point] : in2[point];
      }
    }
  }
};

/*static*/ void WhereTask::cpu_variant(TaskContext& context)
{
  where_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { WhereTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
