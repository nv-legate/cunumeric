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

#include "cunumeric/unary/copy.h"
#include "cunumeric/unary/copy_template.inl"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE, int DIM>
struct CopyImplBody<VariantKind::CPU, CODE, DIM> {
  using VAL   = legate_type_of<CODE>;
  using POINT = Point<DIM>;

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, DIM> in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    if (dense) {
      auto outptr = out.ptr(rect);
      auto inptr  = in.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) outptr[idx] = inptr[idx];
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        out[p] = in[p];
      }
    }
  }

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, DIM> in,
                  AccessorRO<POINT, DIM> indirection,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense,
                  bool is_source_indirect) const
  {
    const size_t volume = rect.volume();
    if (dense) {
      auto outptr = out.ptr(rect);
      auto inptr  = in.ptr(rect);
      auto indptr = indirection.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) {
        if (is_source_indirect)
          outptr[idx] = in[indptr[idx]];
        else
          out[indptr[idx]] = inptr[idx];
      }
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        if (is_source_indirect)
          out[p] = in[indirection[p]];
        else
          out[indirection[p]] = in[p];
      }
    }
  }
};

/*static*/ void CopyTask::cpu_variant(TaskContext& context)
{
  copy_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { CopyTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
