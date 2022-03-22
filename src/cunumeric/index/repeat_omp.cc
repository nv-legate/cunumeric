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

#include "cunumeric/index/repeat.h"
#include "cunumeric/index/repeat_template.inl"
#include <omp.h>

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE, int DIM>
struct RepeatImplBody<VariantKind::OMP, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  size_t operator()(Buffer<VAL>& out,
                    const AccessorRO<VAL, DIM>& in,
                    const int64_t repeats,
                    const int32_t axis,
                    const Pitches<DIM - 1>& pitches,
                    const Rect<DIM>& rect) const
  {
    const size_t volume = rect.volume();
    size_t size         = volume * repeats;
    Memory::Kind kind =
      CuNumeric::has_numamem ? Memory::Kind::SOCKET_MEM : Memory::Kind::SYSTEM_MEM;
    out = create_buffer<VAL>(size, kind);
#pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < size; ++idx) {
      size_t p_idx = idx / repeats;
      auto p       = pitches.unflatten(p_idx, rect.lo);
      out[idx]     = in[p];
    }
    return size;
  }

  size_t operator()(Buffer<VAL>& out,
                    const AccessorRO<VAL, DIM>& in,
                    const AccessorRO<int64_t, DIM>& repeats,
                    const int32_t axis,
                    const Pitches<DIM - 1>& pitches,
                    const Rect<DIM>& rect) const
  {
    const size_t volume = rect.volume();
    int64_t size        = 0;
    Memory::Kind kind =
      CuNumeric::has_numamem ? Memory::Kind::SOCKET_MEM : Memory::Kind::SYSTEM_MEM;
    DeferredBuffer<int64_t, 1> offsets(kind, Rect<1>(0, volume - 1));

    {
      DeferredBuffer<int64_t, 1> sizes(kind, Rect<1>(0, volume - 1));
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) {
        auto point = pitches.unflatten(idx, rect.lo);
        sizes[idx] = repeats[point];
      }

      for (auto idx = 0; idx < volume; ++idx) size += sizes[idx];

      offsets[0] = 0;
      for (auto idx = 1; idx < volume; ++idx) offsets[idx] = offsets[idx - 1] + sizes[idx - 1];
    }  // end section

    out = create_buffer<VAL>(size, kind);

#pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < volume; ++idx) {
      auto p          = pitches.unflatten(idx, rect.lo);
      int64_t out_idx = offsets[idx];
      for (size_t r = 0; r < repeats[p]; r++) { out[out_idx + r] = in[p]; }
    }
    return size;
  }
};

/*static*/ void RepeatTask::omp_variant(TaskContext& context)
{
  repeat_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
