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

#include "cunumeric/index/advanced_indexing.h"
#include "cunumeric/index/advanced_indexing_template.inl"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE, int DIM1, int DIM2, bool IS_SET>
struct AdvancedIndexingImplBody<VariantKind::CPU, CODE, DIM1, DIM2, IS_SET> {
  using VAL = legate_type_of<CODE>;

  void compute_output(Buffer<VAL>& out,
                      const AccessorRO<VAL, DIM1>& input,
                      const AccessorRO<bool, DIM2>& index,
                      const Pitches<DIM1 - 1>& pitches_input,
                      const Rect<DIM1>& rect_input,
                      const Pitches<DIM2 - 1>& pitches_index,
                      const Rect<DIM2>& rect_index,
                      int volume) const
  {
    int64_t out_idx = 0;
    for (size_t idx = 0; idx < volume; ++idx) {
      auto p       = pitches_index.unflatten(idx, rect_index.lo);
      auto p_input = pitches_input.unflatten(idx, rect_input.lo);
      if (index[p] == true) {
        out[out_idx] = input[p_input];
        out_idx++;
      }
    }
  }

  void compute_output(Buffer<Point<DIM1>>& out,
                      const AccessorRO<VAL, DIM1>&,
                      const AccessorRO<bool, DIM2>& index,
                      const Pitches<DIM1 - 1>& pitches_input,
                      const Rect<DIM1>& rect_input,
                      const Pitches<DIM2 - 1>& pitches_index,
                      const Rect<DIM2>& rect_index,
                      int volume) const
  {
    int64_t out_idx = 0;
    for (size_t idx = 0; idx < volume; ++idx) {
      auto p       = pitches_index.unflatten(idx, rect_index.lo);
      auto p_input = pitches_input.unflatten(idx, rect_input.lo);
      if (index[p] == true) {
        out[out_idx] = p_input;
        out_idx++;
      }
    }
  }

  template <typename OUT_TYPE>
  size_t operator()(Buffer<OUT_TYPE>& out,
                    const AccessorRO<VAL, DIM1>& input,
                    const AccessorRO<bool, DIM2>& index,
                    const Pitches<DIM1 - 1>& pitches_input,
                    const Rect<DIM1>& rect_input,
                    const Pitches<DIM2 - 1>& pitches_index,
                    const Rect<DIM2>& rect_index) const
  {
#ifdef DEBUG_CUNUMERIC
    // in this case shapes for input and index arrays  should be the same
    assert(rect_input == rect_index);
#endif
    const size_t volume = rect_index.volume();
    size_t size         = 0;
    for (size_t idx = 0; idx < volume; ++idx) {
      auto p = pitches_index.unflatten(idx, rect_index.lo);
      if (index[p] == true) { size++; }
    }

    out = create_buffer<OUT_TYPE>(size, Memory::Kind::SYSTEM_MEM);

    compute_output(out, input, index, pitches_input, rect_input, pitches_index, rect_index, volume);
    return size;
  }
};

/*static*/ void AdvancedIndexingTask::cpu_variant(TaskContext& context)
{
  advanced_indexing_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  AdvancedIndexingTask::register_variants();
}
}  // namespace

}  // namespace cunumeric
