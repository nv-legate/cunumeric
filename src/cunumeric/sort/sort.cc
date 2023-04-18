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

#include <cstring>
#include <sstream>

#include "cunumeric/sort/sort.h"
#include "cunumeric/sort/sort_cpu.inl"
#include "cunumeric/sort/sort_template.inl"

#include <functional>
#include <numeric>

namespace cunumeric {

using namespace legate;

template <Type CODE, int32_t DIM>
struct SortImplBody<VariantKind::CPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(const Array& input_array,
                  Array& output_array,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t volume,
                  const size_t segment_size_l,
                  const size_t segment_size_g,
                  const bool argsort,
                  const bool stable,
                  const bool is_index_space,
                  const size_t local_rank,
                  const size_t num_ranks,
                  const size_t num_sort_ranks,
                  const std::vector<comm::Communicator>& comms)
  {
    SortImplBodyCpu<CODE, DIM>()(input_array,
                                 output_array,
                                 pitches,
                                 rect,
                                 volume,
                                 segment_size_l,
                                 segment_size_g,
                                 argsort,
                                 stable,
                                 is_index_space,
                                 local_rank,
                                 num_ranks,
                                 num_sort_ranks,
                                 thrust::host,
                                 comms);
  }
};

/*static*/ void SortTask::cpu_variant(TaskContext& context)
{
  sort_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  auto options = legate::VariantOptions{}.with_concurrent(true);
  SortTask::register_variants(
    {{LEGATE_CPU_VARIANT, options}, {LEGATE_GPU_VARIANT, options}, {LEGATE_OMP_VARIANT, options}});
}
}  // namespace

}  // namespace cunumeric
