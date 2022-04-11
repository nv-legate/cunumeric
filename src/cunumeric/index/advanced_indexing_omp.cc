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
#include "cunumeric/omp_help.h"
#include <omp.h>

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE, int DIM1, int DIM2, bool IS_SET>
struct AdvancedIndexingImplBody<VariantKind::OMP, CODE, DIM1, DIM2, IS_SET> {
  using VAL = legate_type_of<CODE>;

  void compute_output(Buffer<VAL>& out,
                      const AccessorRO<VAL, DIM1>& input,
                      const AccessorRO<bool, DIM2>& index,
                      const Pitches<DIM1 - 1>& pitches_input,
                      const Rect<DIM1>& rect_input,
                      const Pitches<DIM2 - 1>& pitches_index,
                      const Rect<DIM2>& rect_index,
                      int volume,
                      int64_t out_idx) const
  {
#pragma omp for schedule(static)
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
                      int volume,
                      int64_t out_idx) const
  {
#pragma omp for schedule(static)
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
    const size_t volume    = rect_index.volume();
    const auto max_threads = omp_get_max_threads();
    int64_t size           = 0;
    ThreadLocalStorage<int64_t> offsets(max_threads);

    {
      ThreadLocalStorage<int64_t> sizes(max_threads);
      for (auto idx = 0; idx < max_threads; ++idx) sizes[idx] = 0;
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (size_t idx = 0; idx < volume; ++idx) {
          auto point = pitches_index.unflatten(idx, rect_index.lo);
          if (index[point] == true) sizes[tid] += 1;
        }
      }

      for (auto idx = 0; idx < max_threads; ++idx) size += sizes[idx];

      offsets[0] = 0;
      for (auto idx = 1; idx < max_threads; ++idx) offsets[idx] = offsets[idx - 1] + sizes[idx - 1];
    }

    Memory::Kind kind =
      CuNumeric::has_numamem ? Memory::Kind::SOCKET_MEM : Memory::Kind::SYSTEM_MEM;
    out = create_buffer<OUT_TYPE>(size, kind);

#pragma omp parallel
    {
      const int tid   = omp_get_thread_num();
      int64_t out_idx = offsets[tid];
      compute_output(
        out, input, index, pitches_input, rect_input, pitches_index, rect_index, volume, out_idx);
    }

    return size;
  }
};

/*static*/ void AdvancedIndexingTask::omp_variant(TaskContext& context)
{
  advanced_indexing_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
