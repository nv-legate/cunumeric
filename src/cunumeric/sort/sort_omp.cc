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

#include <parallel/algorithm>
#include <omp.h>

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE, int32_t DIM>
struct SortImplBody<VariantKind::OMP, false, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void std_sort_omp(const VAL* inptr, VAL* outptr, const size_t volume, const size_t sort_dim_size)
  {
    std::copy(inptr, inptr + volume, outptr);
    if (volume / sort_dim_size > omp_get_max_threads() / 2)  // TODO fine tune
    {
#pragma omp do schedule(dynamic)
      for (uint32_t start_idx = 0; start_idx < volume; start_idx += sort_dim_size) {
        std::stable_sort(outptr + start_idx, outptr + start_idx + sort_dim_size);
      }
    } else {
      for (uint32_t start_idx = 0; start_idx < volume; start_idx += sort_dim_size) {
        __gnu_parallel::stable_sort(outptr + start_idx, outptr + start_idx + sort_dim_size);
      }
    }
  }

  void operator()(AccessorRO<VAL, DIM> input,
                  AccessorWO<VAL, DIM> output,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const bool dense,
                  const size_t volume,
                  const bool argsort,
                  const Legion::DomainPoint global_shape,
                  const bool is_index_space,
                  const Legion::DomainPoint index_point,
                  const Legion::Domain domain)
  {
#ifdef DEBUG_CUNUMERIC
    std::cout << "CPU(" << getRank(domain, index_point) << "): local size = " << volume
              << ", dist. = " << is_index_space << ", index_point = " << index_point
              << ", domain/volume = " << domain << "/" << domain.get_volume()
              << ", dense = " << dense << std::endl;
#endif
    const size_t sort_dim_size = global_shape[DIM - 1];
    assert(!is_index_space || DIM > 1);  // not implemented for now
    if (dense) {
      std_sort_omp(input.ptr(rect), output.ptr(rect), volume, sort_dim_size);
    } else {
      // compute contiguous memory block
      int contiguous_elements = 1;
      for (int i = DIM - 1; i >= 0; i--) {
        auto diff = 1 + rect.hi[i] - rect.lo[i];
        contiguous_elements *= diff;
        if (diff < global_shape[i]) { break; }
      }

      uint64_t elements_processed = 0;
      while (elements_processed < volume) {
        Legion::Point<DIM> start_point = pitches.unflatten(elements_processed, rect.lo);
        std_sort_omp(
          input.ptr(start_point), output.ptr(start_point), contiguous_elements, sort_dim_size);
        elements_processed += contiguous_elements;
      }
    }
  }
};

template <LegateTypeCode CODE, int32_t DIM>
struct SortImplBody<VariantKind::OMP, true, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void std_argsort(const VAL* inptr,
                   int32_t* outptr,
                   const size_t volume,
                   const size_t sort_dim_size)
  {
    for (uint64_t start_idx = 0; start_idx < volume; start_idx += sort_dim_size) {
      int32_t* segmentKeys     = outptr + start_idx;
      const VAL* segmentValues = inptr + start_idx;
      std::iota(outptr + start_idx, outptr + start_idx + sort_dim_size, 0);
      std::stable_sort(
        segmentKeys, segmentKeys + sort_dim_size, [segmentValues](int32_t i1, int32_t i2) {
          return segmentValues[i1] < segmentValues[i2];
        });
    }
  }

  void std_argsort_omp(const VAL* inptr,
                       int32_t* outptr,
                       const size_t volume,
                       const size_t sort_dim_size)
  {
    if (volume / sort_dim_size > omp_get_max_threads() / 2)  // TODO fine tune
    {
#pragma omp do schedule(dynamic)
      for (uint32_t start_idx = 0; start_idx < volume; start_idx += sort_dim_size) {
        int32_t* segmentKeys     = outptr + start_idx;
        const VAL* segmentValues = inptr + start_idx;
        std::iota(segmentKeys, segmentKeys + sort_dim_size, 0);
        std::stable_sort(
          segmentKeys, segmentKeys + sort_dim_size, [segmentValues](int32_t i1, int32_t i2) {
            return segmentValues[i1] < segmentValues[i2];
          });
      }
    } else {
      for (uint32_t start_idx = 0; start_idx < volume; start_idx += sort_dim_size) {
        int32_t* segmentKeys     = outptr + start_idx;
        const VAL* segmentValues = inptr + start_idx;
        std::iota(segmentKeys, segmentKeys + sort_dim_size, 0);
        __gnu_parallel::stable_sort(
          segmentKeys, segmentKeys + sort_dim_size, [segmentValues](int32_t i1, int32_t i2) {
            return segmentValues[i1] < segmentValues[i2];
          });
      }
    }
  }

  void operator()(AccessorRO<VAL, DIM> input,
                  AccessorWO<int32_t, DIM> output,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const bool dense,
                  const size_t volume,
                  const bool argsort,
                  const Legion::DomainPoint global_shape,
                  const bool is_index_space,
                  const Legion::DomainPoint index_point,
                  const Legion::Domain domain)
  {
#ifdef DEBUG_CUNUMERIC
    std::cout << "CPU(" << getRank(domain, index_point) << "): local size = " << volume
              << ", dist. = " << is_index_space << ", index_point = " << index_point
              << ", domain/volume = " << domain << "/" << domain.get_volume()
              << ", dense = " << dense << std::endl;
#endif
    const size_t sort_dim_size = global_shape[DIM - 1];
    assert(!is_index_space || DIM > 1);  // not implemented for now
    if (dense) {
      std_argsort_omp(input.ptr(rect), output.ptr(rect), volume, sort_dim_size);
    } else {
      // compute contiguous memory block
      int contiguous_elements = 1;
      for (int i = DIM - 1; i >= 0; i--) {
        auto diff = 1 + rect.hi[i] - rect.lo[i];
        contiguous_elements *= diff;
        if (diff < global_shape[i]) { break; }
      }

      uint64_t elements_processed = 0;
      while (elements_processed < volume) {
        Legion::Point<DIM> start_point = pitches.unflatten(elements_processed, rect.lo);
        std_argsort_omp(
          input.ptr(start_point), output.ptr(start_point), contiguous_elements, sort_dim_size);
        elements_processed += contiguous_elements;
      }
    }
  }
};

/*static*/ void SortTask::omp_variant(TaskContext& context)
{
  sort_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
