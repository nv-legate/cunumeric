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

#include "cunumeric/sort/sort.h"
#include "cunumeric/sort/sort_template.inl"

#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;

template <LegateTypeCode CODE, int32_t DIM>
struct SortImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(VAL* inptr,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t volume,
                  const size_t sort_dim_size,
                  bool is_index_space,
                  Legion::DomainPoint index_point,
                  Legion::Domain domain)
  {
#ifdef DEBUG_CUNUMERIC
    std::cout << "GPU(" << index_point[0] << "): local size = " << volume
              << ", dist. = " << is_index_space << ", index_point = " << index_point
              << ", domain/volume = " << domain << "/" << domain.get_volume() << std::endl;
#endif

    thrust::device_ptr<VAL> dev_ptr(inptr);
    for (uint32_t start_idx = 0; start_idx < volume; start_idx += sort_dim_size) {
      thrust::stable_sort(dev_ptr + start_idx, dev_ptr + start_idx + sort_dim_size);
    }

    // in case of distributed data we need to switch to sample sort
    if (is_index_space && DIM == 1) {
      // not implemented yet
      assert(false);
    }
  }
};

/*static*/ void SortTask::gpu_variant(TaskContext& context)
{
  sort_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
