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

#include "cunumeric/index/repeat.h"
#include "cunumeric/index/repeat_template.inl"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE, int DIM>
struct RepeatImplBody<VariantKind::OMP, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(const AccessorWO<VAL, DIM>& out,
                  const AccessorRO<VAL, DIM>& in,
                  const int64_t repeats,
                  const int32_t axis,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect) const
  {
    const size_t volume = rect.volume();

#pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < volume; ++idx) {
      auto p            = pitches.unflatten(idx, rect.lo);
      int64_t input_idx = p[axis] / repeats;
      auto in_p         = p;
      in_p[axis]        = input_idx;
      out[p]            = in[in_p];
    }
  }

  void operator()(const AccessorWO<VAL, DIM>& out,
                  const AccessorRO<VAL, DIM>& in,
                  const AccessorRO<int64_t, 1>& repeats,
                  const int32_t axis,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  size_t repeats_size) const
  {
    const size_t volume = rect.volume();
    std::vector<int64_t> offsets;
    for (size_t r = 0; r <= repeats_size; r++) {
      for (size_t i = 0; i < repeats[r]; i++) { offsets.push_back(r); }
    }
#pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < volume; ++idx) {
      auto p     = pitches.unflatten(idx, rect.lo);
      auto in_p  = p;
      in_p[axis] = offsets[p[axis]];
      out[p]     = in[in_p];
    }
  }
};

/*static*/ void RepeatTask::omp_variant(TaskContext& context)
{
  repeat_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
