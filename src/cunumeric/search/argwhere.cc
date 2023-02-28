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

#include "cunumeric/search/argwhere.h"
#include "cunumeric/search/argwhere_template.inl"

namespace cunumeric {

using namespace legate;

template <LegateTypeCode CODE, int DIM>
struct ArgWhereImplBody<VariantKind::CPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(Array& out_array,
                  AccessorRO<VAL, DIM> input,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  size_t volume) const
  {
    int64_t size = 0;
    for (size_t idx = 0; idx < volume; ++idx) {
      auto in_p = pitches.unflatten(idx, rect.lo);
      size += input[in_p] != VAL(0);
    }

    auto out = out_array.create_output_buffer<int64_t, 2>(Point<2>(size, DIM), true);

    size_t out_idx = 0;
    for (size_t idx = 0; idx < volume; ++idx) {
      auto in_p = pitches.unflatten(idx, rect.lo);

      if (input[in_p] != VAL(0)) {
        for (int i = 0; i < DIM; ++i) { out[Point<2>(out_idx, i)] = in_p[i]; }
        out_idx++;
      }
    }

    assert(size == out_idx);
  }
};

/*static*/ void ArgWhereTask::cpu_variant(TaskContext& context)
{
  argwhere_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { ArgWhereTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
