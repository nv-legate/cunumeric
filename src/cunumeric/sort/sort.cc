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

#include "cunumeric/sort/sort.h"
#include "cunumeric/sort/sort_template.inl"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <numeric>

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE, int32_t DIM>
struct SortImplBody<VariantKind::CPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  // sorts inptr in-place, if argptr not nullptr it returns sort indices
  void thrust_local_sort_inplace(VAL* inptr,
                                 int64_t* argptr,
                                 const size_t volume,
                                 const size_t sort_dim_size,
                                 const bool stable_argsort)
  {
    if (argptr == nullptr) {
      // sort (in place)
      for (size_t start_idx = 0; start_idx < volume; start_idx += sort_dim_size) {
        thrust::sort(thrust::host, inptr + start_idx, inptr + start_idx + sort_dim_size);
      }
    } else {
      // argsort
      for (uint64_t start_idx = 0; start_idx < volume; start_idx += sort_dim_size) {
        int64_t* segmentValues = argptr + start_idx;
        VAL* segmentKeys       = inptr + start_idx;
        std::iota(segmentValues, segmentValues + sort_dim_size, 0);  // init
        if (stable_argsort) {
          thrust::stable_sort_by_key(
            thrust::host, segmentKeys, segmentKeys + sort_dim_size, segmentValues);
        } else {
          thrust::sort_by_key(
            thrust::host, segmentKeys, segmentKeys + sort_dim_size, segmentValues);
        }
      }
    }
  }

  void operator()(const Array& input_array,
                  Array& output_array,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t volume,
                  const size_t sort_dim_size,
                  const bool argsort,
                  const bool stable,
                  const bool is_index_space,
                  const size_t local_rank,
                  const size_t num_ranks,
                  const std::vector<comm::Communicator>& comms)
  {
    auto input = input_array.read_accessor<VAL, DIM>(rect);
    assert(input.accessor.is_dense_row_major(rect));
    assert(!is_index_space || DIM > 1);

    if (argsort) {
      // make copy of the input
      auto dense_input_copy = create_buffer<VAL>(volume);
      {
        auto* src = input.ptr(rect.lo);
        std::copy(src, src + volume, dense_input_copy.ptr(0));
      }

      AccessorWO<int64_t, DIM> output = output_array.write_accessor<int64_t, DIM>(rect);
      assert(output.accessor.is_dense_row_major(rect));

      // sort data in place
      thrust_local_sort_inplace(
        dense_input_copy.ptr(0), output.ptr(rect.lo), volume, sort_dim_size, stable);

    } else {
      AccessorWO<VAL, DIM> output = output_array.write_accessor<VAL, DIM>(rect);
      assert(output.accessor.is_dense_row_major(rect));

      // init output values
      auto* src    = input.ptr(rect.lo);
      auto* target = output.ptr(rect.lo);
      if (src != target) std::copy(src, src + volume, target);

      // sort data in place
      thrust_local_sort_inplace(target, nullptr, volume, sort_dim_size, stable);
    }
  }
};

/*static*/ void SortTask::cpu_variant(TaskContext& context)
{
  sort_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { SortTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
