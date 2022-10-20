/* Copyright 2022 NVIDIA Corporation
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

#include "cunumeric/index/wrap.h"
#include "cunumeric/index/wrap_template.inl"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <int DIM>
struct WrapImplBody<VariantKind::CPU, DIM> {
  template <typename IND>
  void operator()(const AccessorWO<Point<DIM>, 1>& out,
                  const Pitches<0>& pitches_out,
                  const Rect<1>& out_rect,
                  const Pitches<DIM - 1>& pitches_in,
                  const Rect<DIM>& in_rect,
                  const bool dense,
                  const IND& indices) const
  {
    const int64_t start  = out_rect.lo[0];
    const int64_t end    = out_rect.hi[0];
    const auto in_volume = in_rect.volume();
    if (dense) {
      auto outptr = out.ptr(out_rect);
      for (int64_t i = start; i <= end; i++) {
        check_idx(i, in_volume, indices);
        const int64_t input_idx = compute_idx(i, in_volume, indices);
        auto point              = pitches_in.unflatten(input_idx, in_rect.lo);
        outptr[i - start]       = point;
      }
    } else {
      for (int64_t i = start; i <= end; i++) {
        check_idx(i, in_volume, indices);
        const int64_t input_idx = compute_idx(i, in_volume, indices);
        auto point              = pitches_in.unflatten(input_idx, in_rect.lo);
        out[i]                  = point;
      }
    }  // else
  }
};

/*static*/ void WrapTask::cpu_variant(TaskContext& context)
{
  wrap_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { WrapTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
