/* Copyright 2021 NVIDIA Corporation
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

#include "numpy/search/nonzero.h"
#include "numpy/search/nonzero_template.inl"

namespace legate {
namespace numpy {

using namespace Legion;

template <LegateTypeCode CODE, int32_t DIM>
struct NonzeroImplBody<VariantKind::CPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  size_t operator()(const AccessorRO<VAL, DIM>& in,
                    const Pitches<DIM - 1>& pitches,
                    const Rect<DIM>& rect,
                    const size_t volume,
                    std::vector<Buffer<int64_t>>& results)
  {
    int64_t size = 0;

    for (size_t idx = 0; idx < volume; ++idx) {
      auto point = pitches.unflatten(idx, rect.lo);
      size += in[point] != VAL(0);
    }

    for (auto& result : results) result = create_buffer<int64_t>(size, Memory::Kind::SYSTEM_MEM);

    int64_t out_idx = 0;
    for (size_t idx = 0; idx < volume; ++idx) {
      auto point = pitches.unflatten(idx, rect.lo);
      if (in[point] == VAL(0)) continue;
      for (int32_t dim = 0; dim < DIM; ++dim) results[dim][out_idx] = point[dim];
      ++out_idx;
    }
    assert(size == out_idx);

    return size;
  }
};

/*static*/ void NonzeroTask::cpu_variant(TaskContext& context)
{
  nonzero_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { NonzeroTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
