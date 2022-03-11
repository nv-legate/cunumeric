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

#include "cunumeric/set/unique.h"
#include "cunumeric/set/unique_template.inl"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE, int32_t DIM>
struct UniqueImplBody<VariantKind::CPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  std::pair<Buffer<VAL>, size_t> operator()(const AccessorRO<VAL, DIM>& in,
                                            const Pitches<DIM - 1>& pitches,
                                            const Rect<DIM>& rect,
                                            const size_t volume,
                                            const std::vector<comm::Communicator>& comms,
                                            const DomainPoint& point,
                                            const Domain& launch_domain)
  {
    std::set<VAL> dedup_set;

    for (size_t idx = 0; idx < volume; ++idx) {
      auto p = pitches.unflatten(idx, rect.lo);
      dedup_set.insert(in[p]);
    }

    size_t size = dedup_set.size();
    size_t pos  = 0;
    auto result = create_buffer<VAL>(size);

    for (auto e : dedup_set) result[pos++] = e;

    return std::make_pair(result, size);
  }
};

/*static*/ void UniqueTask::cpu_variant(TaskContext& context)
{
  unique_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { UniqueTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
