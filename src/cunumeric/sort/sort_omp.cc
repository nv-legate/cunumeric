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
#include "cunumeric/sort/sort_cpu.inl"
#include "cunumeric/sort/sort_template.inl"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>
#include <numeric>
#include <omp.h>

namespace cunumeric {

using namespace Legion;
using namespace legate;

// sorts inptr in-place, if argptr not nullptr it returns sort indices
template <typename VAL>
void thrust_local_sort_inplace(VAL* inptr,
                               int64_t* argptr,
                               const size_t volume,
                               const size_t sort_dim_size,
                               const bool stable_argsort)
{
  if (argptr == nullptr) {
    // sort (in place)
    for (size_t start_idx = 0; start_idx < volume; start_idx += sort_dim_size) {
      thrust::sort(thrust::omp::par, inptr + start_idx, inptr + start_idx + sort_dim_size);
    }
  } else {
    // argsort
    for (uint64_t start_idx = 0; start_idx < volume; start_idx += sort_dim_size) {
      int64_t* segmentValues = argptr + start_idx;
      VAL* segmentKeys       = inptr + start_idx;
      if (stable_argsort) {
        thrust::stable_sort_by_key(
          thrust::omp::par, segmentKeys, segmentKeys + sort_dim_size, segmentValues);
      } else {
        thrust::sort_by_key(
          thrust::omp::par, segmentKeys, segmentKeys + sort_dim_size, segmentValues);
      }
    }
  }
}

template <LegateTypeCode CODE, int32_t DIM>
struct SortImplBody<VariantKind::OMP, CODE, DIM> {
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
    auto input = input_array.read_accessor<VAL, DIM>(rect);

    // we allow empty domains for distributed sorting
    assert(rect.empty() || input.accessor.is_dense_row_major(rect));

    bool is_unbound_1d_storage = output_array.is_output_store();
    bool need_distributed_sort = segment_size_l != segment_size_g || is_unbound_1d_storage;
    bool rebalance             = !is_unbound_1d_storage;
    assert(DIM == 1 || !is_unbound_1d_storage);

    // initialize sort pointers
    SortPiece<VAL> local_sorted;
    int64_t* indices_ptr = nullptr;
    VAL* values_ptr      = nullptr;
    if (argsort) {
      // make a buffer for input
      auto input_copy     = create_buffer<VAL>(volume);
      local_sorted.values = input_copy;
      values_ptr          = input_copy.ptr(0);

      // initialize indices
      if (need_distributed_sort) {
        auto indices_buffer  = create_buffer<int64_t>(volume);
        indices_ptr          = indices_buffer.ptr(0);
        local_sorted.indices = indices_buffer;
        local_sorted.size    = volume;
      } else {
        AccessorWO<int64_t, DIM> output = output_array.write_accessor<int64_t, DIM>(rect);
        assert(rect.empty() || output.accessor.is_dense_row_major(rect));
        indices_ptr = output.ptr(rect.lo);
      }
      size_t offset = rect.lo[DIM - 1];
      if (volume > 0) {
        if (DIM == 1) {
          thrust::sequence(thrust::omp::par, indices_ptr, indices_ptr + volume, offset);
        } else {
          thrust::transform(thrust::omp::par,
                            thrust::make_counting_iterator<int64_t>(0),
                            thrust::make_counting_iterator<int64_t>(volume),
                            thrust::make_constant_iterator<int64_t>(segment_size_l),
                            indices_ptr,
                            modulusWithOffset(offset));
        }
      }
    } else {
      // initialize output
      if (need_distributed_sort) {
        auto input_copy      = create_buffer<VAL>(volume);
        values_ptr           = input_copy.ptr(0);
        local_sorted.values  = input_copy;
        local_sorted.indices = create_buffer<int64_t>(0);
        local_sorted.size    = volume;
      } else {
        AccessorWO<VAL, DIM> output = output_array.write_accessor<VAL, DIM>(rect);
        assert(rect.empty() || output.accessor.is_dense_row_major(rect));
        values_ptr = output.ptr(rect.lo);
      }
    }

    if (volume > 0) {
      // sort data (locally)
      auto* src = input.ptr(rect.lo);
      if (src != values_ptr) std::copy(src, src + volume, values_ptr);
      thrust_local_sort_inplace(values_ptr, indices_ptr, volume, segment_size_l, stable);
    }

    if (need_distributed_sort) {
      if (is_index_space) {
        assert(is_index_space || is_unbound_1d_storage);
        std::vector<size_t> sort_ranks(num_sort_ranks);
        size_t rank_group = local_rank / num_sort_ranks;
        for (int r = 0; r < num_sort_ranks; ++r) sort_ranks[r] = rank_group * num_sort_ranks + r;

        void* output_ptr = nullptr;
        // in case the storage *is NOT* unbound -- we provide a target pointer
        // in case the storage *is* unbound -- the result will be appended to output_array
        if (volume > 0 && !is_unbound_1d_storage) {
          if (argsort) {
            auto output = output_array.write_accessor<int64_t, DIM>(rect);
            assert(output.accessor.is_dense_row_major(rect));
            output_ptr = static_cast<void*>(output.ptr(rect.lo));
          } else {
            auto output = output_array.write_accessor<VAL, DIM>(rect);
            assert(output.accessor.is_dense_row_major(rect));
            output_ptr = static_cast<void*>(output.ptr(rect.lo));
          }
        }

        sample_sort_nd<CODE>(local_sorted,
                             output_array,
                             output_ptr,
                             local_rank,
                             num_ranks,
                             segment_size_g,
                             local_rank % num_sort_ranks,
                             num_sort_ranks,
                             sort_ranks.data(),
                             segment_size_l,
                             rebalance,
                             argsort,
                             comms[0].get<comm::coll::CollComm>());
      } else {
        // edge case where we have an unbound store but only 1 CPU was assigned with the task
        if (argsort) {
          local_sorted.values.destroy();
          output_array.return_data(local_sorted.indices, Point<1>(local_sorted.size));
        } else {
          output_array.return_data(local_sorted.values, Point<1>(local_sorted.size));
        }
      }
    } else if (argsort) {
      // cleanup for non distributed argsort
      local_sorted.values.destroy();
    }
  }
};

/*static*/ void SortTask::omp_variant(TaskContext& context)
{
  sort_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
