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

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE, int DIM>
struct RepeatImplBody<VariantKind::CPU, CODE, DIM> {
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
    out                 = create_buffer<VAL>(size, Memory::Kind::SYSTEM_MEM);

    int64_t out_idx = 0;
    for (size_t idx = 0; idx < volume; ++idx) {
      auto p = pitches.unflatten(idx, rect.lo);
      for (size_t r = 0; r < repeats; r++) {
        out[out_idx] = in[p];
        ++out_idx;
      }
    }
#ifdef CUNUMERIC_DEBUG
    assert(size == out_idx);
#endif
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
    size_t size         = 0;
    for (size_t idx = 0; idx < volume; ++idx) {
      auto point = pitches.unflatten(idx, rect.lo);
      size += repeats[point];
    }
    out = create_buffer<VAL>(size, Memory::Kind::SYSTEM_MEM);

    int64_t out_idx = 0;
    for (size_t idx = 0; idx < volume; ++idx) {
      auto p = pitches.unflatten(idx, rect.lo);
      for (size_t r = 0; r < repeats[p]; r++) {
        out[out_idx] = in[p];
        ++out_idx;
      }
    }
#ifdef CUNUMERIC_DEBUG
    assert(size == out_idx);
#endif
    return size;
  }
};

/*static*/ void RepeatTask::cpu_variant(TaskContext& context)
{
  repeat_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { RepeatTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
