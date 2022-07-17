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
  void operator()(const AccessorWO<Point<DIM>, 1>& out,
                  const Rect<1>& out_rect,
                  const Pitches<DIM - 1>& pitches_in,
                  const Rect<DIM> rect_in) const
  {
    size_t start = out_rect.lo[0];
    size_t end   = out_rect.hi[0];
    size_t count = 0;
    auto outptr  = out.ptr(out_rect);
    for (size_t i = start; i <= end; i++) {
      auto point    = pitches_in.unflatten(start, rect_in.lo);
      outptr[count] = point;
      count++;
    }
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
