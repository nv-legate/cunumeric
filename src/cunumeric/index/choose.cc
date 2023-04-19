/* Copyright 2021-2022 NVIDIA Corporation
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

#include "cunumeric/index/choose.h"
#include "cunumeric/index/choose_template.inl"

namespace cunumeric {

using namespace legate;

template <Type::Code CODE, int DIM>
struct ChooseImplBody<VariantKind::CPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(const AccessorWO<VAL, DIM>& out,
                  const AccessorRO<int64_t, DIM>& index_arr,
                  const std::vector<AccessorRO<VAL, DIM>>& choices,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    if (dense) {
      auto outptr   = out.ptr(rect);
      auto indexptr = index_arr.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) {
#ifdef DEBUG_CUNUMERIC
        assert(indexptr[idx] < choices.size());
#endif
        auto chptr  = choices[indexptr[idx]].ptr(rect);
        outptr[idx] = chptr[idx];
      }
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
#ifdef DEBUG_CUNUMERIC
        assert(index_arr[p] < choices.size());
#endif
        out[p] = choices[index_arr[p]][p];
      }
    }
  }
};

/*static*/ void ChooseTask::cpu_variant(TaskContext& context)
{
  choose_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { ChooseTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
