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

#include "cunumeric/sort/sort.h"
#include "cunumeric/sort/sort_template.inl"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <numeric>
#include <omp.h>

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE, int32_t DIM>
struct SortImplBody<VariantKind::OMP, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  // sorts inptr in-place, if argptr not nullptr it returns sort indices
  void thrust_local_sort_inplace(VAL* inptr,
                                 int64_t* argptr,
                                 const size_t volume,
                                 const size_t sort_dim_size)
  {
    if (argptr == nullptr) {
      // sort (in place)
#pragma omp parallel for
      for (size_t start_idx = 0; start_idx < volume; start_idx += sort_dim_size) {
        thrust::stable_sort(thrust::host, inptr + start_idx, inptr + start_idx + sort_dim_size);
      }
    } else {
      // argsort
#pragma omp parallel for
      for (uint64_t start_idx = 0; start_idx < volume; start_idx += sort_dim_size) {
        int64_t* segmentValues = argptr + start_idx;
        VAL* segmentKeys       = inptr + start_idx;
        std::iota(segmentValues, segmentValues + sort_dim_size, 0);  // init
        thrust::stable_sort_by_key(
          thrust::host, segmentKeys, segmentKeys + sort_dim_size, segmentValues);
      }
    }
  }

  void operator()(const Array& input_array,
                  Array& output_array,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t volume,
                  const bool argsort,
                  const Legion::DomainPoint global_shape,
                  const bool is_index_space,
                  const Legion::DomainPoint index_point,
                  const Legion::Domain domain)
  {
    AccessorRO<VAL, DIM> input = input_array.read_accessor<VAL, DIM>(rect);

    bool dense = input.accessor.is_dense_row_major(rect);

#ifdef DEBUG_CUNUMERIC
    std::cout << "OMP(" << getRank(domain, index_point) << "): local size = " << volume
              << ", dist. = " << is_index_space << ", index_point = " << index_point
              << ", domain/volume = " << domain << "/" << domain.get_volume()
              << ", dense = " << dense << ", argsort. = " << argsort << std::endl;
#endif

    const size_t sort_dim_size = global_shape[DIM - 1];
    assert(!is_index_space || DIM > 1);  // not implemented for now

    // make a copy of the input
    auto dense_input_copy = create_buffer<VAL>(volume, Legion::Memory::Kind::SOCKET_MEM);
    if (dense) {
      auto* src = input.ptr(rect.lo);
      std::copy(src, src + volume, dense_input_copy.ptr(0));
    } else {
      auto* target = dense_input_copy.ptr(0);
      for (size_t offset = 0; offset < volume; ++offset) {
        auto point     = pitches.unflatten(offset, rect.lo);
        target[offset] = input[rect.lo + point];
      }
    }

    // we need a buffer for argsort
    auto indices_buffer =
      create_buffer<int64_t>(argsort ? volume : 0, Legion::Memory::Kind::SOCKET_MEM);

    // sort data
    thrust_local_sort_inplace(
      dense_input_copy.ptr(0), argsort ? indices_buffer.ptr(0) : nullptr, volume, sort_dim_size);

    // copy back data (we assume output partition to be aliged to input!)
    if (dense) {
      if (argsort) {
        AccessorWO<int64_t, DIM> output = output_array.write_accessor<int64_t, DIM>(rect);
        std::copy(indices_buffer.ptr(0), indices_buffer.ptr(0) + volume, output.ptr(rect.lo));
      } else {
        AccessorWO<VAL, DIM> output = output_array.write_accessor<VAL, DIM>(rect);
        std::copy(dense_input_copy.ptr(0), dense_input_copy.ptr(0) + volume, output.ptr(rect.lo));
      }
    } else {
      if (argsort) {
        AccessorWO<int64_t, DIM> output = output_array.write_accessor<int64_t, DIM>(rect);
        auto* source                    = indices_buffer.ptr(0);
        for (size_t offset = 0; offset < volume; ++offset) {
          auto point              = pitches.unflatten(offset, rect.lo);
          output[rect.lo + point] = source[offset];
        }
      } else {
        AccessorWO<VAL, DIM> output = output_array.write_accessor<VAL, DIM>(rect);
        auto* source                = dense_input_copy.ptr(0);
        for (size_t offset = 0; offset < volume; ++offset) {
          auto point              = pitches.unflatten(offset, rect.lo);
          output[rect.lo + point] = source[offset];
        }
      }
    }
  }
};

/*static*/ void SortTask::omp_variant(TaskContext& context)
{
  sort_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
