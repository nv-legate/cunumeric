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

#include "cunumeric/set/unique_reduce.h"
#include "cunumeric/set/unique_reduce_template.inl"

namespace cunumeric {

using namespace legate;

template <LegateTypeCode CODE>
struct UniqueReduceImplBody<VariantKind::CPU, CODE> {
  using VAL = legate_type_of<CODE>;

  void operator()(Array& output, const std::vector<std::pair<AccessorRO<VAL, 1>, Rect<1>>>& inputs)
  {
    std::set<VAL> dedup_set;

    for (auto& pair : inputs) {
      auto& input = pair.first;
      auto& shape = pair.second;
      for (coord_t idx = shape.lo[0]; idx <= shape.hi[0]; ++idx) dedup_set.insert(input[idx]);
    }

    size_t size = dedup_set.size();
    size_t pos  = 0;
    auto result = output.create_output_buffer<VAL, 1>(Point<1>(size), true);

    for (auto e : dedup_set) result[pos++] = e;
  }
};

/*static*/ void UniqueReduceTask::cpu_variant(TaskContext& context)
{
  unique_reduce_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  UniqueReduceTask::register_variants();
}
}  // namespace

}  // namespace cunumeric
