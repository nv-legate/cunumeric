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

template <LegateTypeCode CODE, int DIM, typename OUT_TYPE>
struct AdvancedIndexingImplBody<VariantKind::CPU, CODE, DIM, OUT_TYPE> {
  using VAL = legate_type_of<CODE>;

  void compute_output(Buffer<VAL, DIM>& out,
                      const AccessorRO<VAL, DIM>& input,
                      const AccessorRO<bool, DIM>& index,
                      const Pitches<DIM - 1>& pitches,
                      const Rect<DIM>& rect,
                      const int volume,
                      const int key_dim,
                      const int skip_size) const
  {
    size_t out_idx = 0;
    for (size_t idx = 0; idx < volume; ++idx) {
      auto p = pitches.unflatten(idx, rect.lo);
      Point<DIM> out_p;
      for (size_t i = 0; i < key_dim - 1; i++) { out_p[i] = 0; }
      out_p[key_dim - 1] = out_idx;
      for (size_t i = key_dim; i < DIM; i++) { out_p[i] = p[i]; }
      if (index[p] == true) {
        out[out_p] = input[p];
        if ((idx != 0 and idx % skip_size == 0) or (skip_size == 1)) out_idx++;
      }
    }
  }

  void compute_output(Buffer<Point<DIM>, DIM>& out,
                      const AccessorRO<VAL, DIM>&,
                      const AccessorRO<bool, DIM>& index,
                      const Pitches<DIM - 1>& pitches,
                      const Rect<DIM>& rect,
                      const int volume,
                      const int key_dim,
                      const int skip_size) const
  {
    size_t out_idx = 0;
    for (size_t idx = 0; idx < volume; ++idx) {
      auto p = pitches.unflatten(idx, rect.lo);
      Point<DIM> out_p;
      for (size_t i = 0; i < key_dim - 1; i++) { out_p[i] = 0; }
      out_p[key_dim - 1] = out_idx;
      for (size_t i = key_dim; i < DIM; i++) { out_p[i] = p[i]; }
      if (index[p] == true) {
        out[out_p] = p;
        if ((idx != 0 and idx % skip_size == 0) or (skip_size == 1)) out_idx++;
      }
    }
  }

  size_t operator()(Array& out_arr,
                    const AccessorRO<VAL, DIM>& input,
                    const AccessorRO<bool, DIM>& index,
                    const Pitches<DIM - 1>& pitches,
                    const Rect<DIM>& rect,
                    const size_t key_dim) const
  {
    Point<DIM> extends;
    size_t skip_size = 1;
    for (int i = key_dim; i < DIM; i++) {
      auto diff  = 1 + rect.hi[i] - rect.lo[i];
      extends[i] = diff;
      if (diff != 0) skip_size *= diff;
    }
    for (int i = 0; i < key_dim - 1; i++) extends[i] = 1;

    const size_t volume = rect.volume();
    size_t size         = 0;
    for (size_t idx = 0; idx < volume; idx += skip_size) {
      auto p = pitches.unflatten(idx, rect.lo);
      if (index[p] == true) { size++; }
    }

    extends[key_dim - 1] = size;

    auto out = out_arr.create_output_buffer<OUT_TYPE, DIM>(extends, true);

    compute_output(out, input, index, pitches, rect, volume, key_dim, skip_size);
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
