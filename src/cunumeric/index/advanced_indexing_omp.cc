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
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE, int DIM, typename OUT_TYPE>
struct AdvancedIndexingImplBody<VariantKind::OMP, CODE, DIM, OUT_TYPE> {
  using VAL = legate_type_of<CODE>;

  void compute_output(Buffer<VAL, DIM>& out,
                      const AccessorRO<VAL, DIM>& input,
                      const AccessorRO<bool, DIM>& index,
                      const Pitches<DIM - 1>& pitches,
                      const Rect<DIM>& rect,
                      const int volume,
                      int64_t out_idx,
                      const int key_dim,
                      const int skip_size) const
  {
#pragma omp for schedule(static)
    for (size_t idx = 0; idx < volume; ++idx) {
      auto p = pitches.unflatten(idx, rect.lo);
      Point<DIM> out_p;
      out_p[0] = out_idx;
      for (int i = key_dim; i < DIM; i++) { out_p[i - key_dim + 1] = p[i]; }
      if (index[p] == true) {
        out[out_p] = input[p];
        if ((idx != 0 and idx % skip_size == 0) or (skip_size == 1)) out_idx++;
      }
    }
  }

  void compute_output(Buffer<Point<DIM>, DIM>& out,
                      const AccessorRO<VAL, DIM>&,
                      const AccessorRO<bool, DIM>& index,
                      const Pitches<DIM - 1>& pitches,
                      const Rect<DIM>& rect,
                      const int volume,
                      int64_t out_idx,
                      const int key_dim,
                      const int skip_size) const
  {
#pragma omp for schedule(static)
    for (size_t idx = 0; idx < volume; ++idx) {
      auto p = pitches.unflatten(idx, rect.lo);
      Point<DIM> out_p;
      out_p[0] = out_idx;
      for (int i = key_dim; i < DIM; i++) { out_p[i - key_dim + 1] = p[i]; }
      if (index[p] == true) {
        out[out_p] = p;
        if ((idx != 0 and idx % skip_size == 0) or (skip_size == 1)) out_idx++;
      }
    }
  }

  void operator()(Array& out_arr,
                  const AccessorRO<VAL, DIM>& input,
                  const AccessorRO<bool, DIM>& index,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const int key_dim) const
  {
    Point<DIM> extends;
    size_t skip_size = 1;
    for (int i = key_dim; i < DIM; i++) {
      auto diff                = 1 + rect.hi[i] - rect.lo[i];
      extends[i - key_dim + 1] = diff;
      if (diff != 0) skip_size *= diff;
    }

    const auto max_threads = omp_get_max_threads();
    const size_t volume    = rect.volume();
    size_t size            = 0;
    ThreadLocalStorage<int64_t> offsets(max_threads);
    {
      ThreadLocalStorage<int64_t> sizes(max_threads);
      thrust::fill(thrust::omp::par, sizes.begin(), sizes.end(), 0);
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (size_t idx = 0; idx < volume; idx += skip_size) {
          auto p = pitches.unflatten(idx, rect.lo);
          if (index[p] == true) { sizes[tid] += 1; }
        }
      }  // end parallel region

      size = thrust::reduce(thrust::omp::par, sizes.begin(), sizes.end(), 0);
      thrust::fill(thrust::omp::par, sizes.begin(), sizes.end(), 0);
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (size_t idx = 0; idx < volume; ++idx) {
          auto p = pitches.unflatten(idx, rect.lo);
          if (index[p] == true) {
            if ((idx != 0 and idx % skip_size == 0) or (skip_size == 1)) sizes[tid] += 1;
          }
        }
      }  // end of parallel
      thrust::exclusive_scan(thrust::omp::par, sizes.begin(), sizes.end(), offsets.begin());
    }  // end scope

    extends[0] = size;

    Memory::Kind kind =
      CuNumeric::has_numamem ? Memory::Kind::SOCKET_MEM : Memory::Kind::SYSTEM_MEM;
    auto out = out_arr.create_output_buffer<OUT_TYPE, DIM>(extends, kind);

#pragma omp parallel
    {
      const int tid   = omp_get_thread_num();
      int64_t out_idx = offsets[tid];
      compute_output(out, input, index, pitches, rect, volume, out_idx, key_dim, skip_size);
    }  // end parallel region
  }
};

/*static*/ void AdvancedIndexingTask::omp_variant(TaskContext& context)
{
  advanced_indexing_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
