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

#include <omp.h>

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <typename VAL>
void merge(VAL* inptr, size_t start_idx, size_t end_idx, VAL* tmp)
{
  const size_t mid  = (end_idx + start_idx) / 2;
  size_t left_idx   = start_idx;
  size_t right_idx  = mid;
  size_t target_idx = start_idx;

  while (left_idx < mid && right_idx < end_idx) {
    if (inptr[left_idx] <= inptr[right_idx]) {
      tmp[target_idx++] = inptr[left_idx++];
    } else {
      tmp[target_idx++] = inptr[right_idx++];
    }
  }

  while (left_idx < mid) { tmp[target_idx++] = inptr[left_idx++]; }
  while (right_idx < end_idx) { tmp[target_idx++] = inptr[right_idx++]; }

  std::copy(tmp + start_idx, tmp + end_idx, inptr + start_idx);
}

// TODO tune
#define SEQUENTIAL_THRESHOLD 1024
#define TASK_THRESHOLD 2048

template <typename VAL>
void merge_sort(VAL* inptr, const size_t start_idx, const size_t end_idx, VAL* tmp)
{
  const size_t size = end_idx - start_idx + 1;
  if (size > SEQUENTIAL_THRESHOLD) {
    const size_t mid = (end_idx + start_idx) / 2;

#pragma omp task shared(inptr, tmp) if (size > TASK_THRESHOLD)
    merge_sort(inptr, start_idx, mid, tmp);
#pragma omp task shared(inptr, tmp) if (size > TASK_THRESHOLD)
    merge_sort(inptr, mid, end_idx, tmp);

#pragma omp taskwait
    merge(inptr, start_idx, end_idx, tmp);
  } else if (size > 1) {
    std::stable_sort(inptr + start_idx, inptr + end_idx);
  }
}

template <LegateTypeCode CODE, int32_t DIM>
struct SortImplBody<VariantKind::OMP, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(VAL* inptr,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t volume,
                  bool is_index_space,
                  Legion::DomainPoint index_point,
                  Legion::Domain domain)
  {
#ifdef DEBUG_CUNUMERIC
    std::cout << "OMP(" << index_point[0] << ":" << omp_get_max_threads() << ":" << omp_get_nested()
              << "): local size = " << volume << ", dist. = " << is_index_space
              << ", index_point = " << index_point << ", domain/volume = " << domain << "/"
              << domain.get_volume() << std::endl;
#endif

    bool nested = omp_get_nested();
    if (!nested) omp_set_nested(1);

    // merge sort
    auto tmp = std::make_unique<VAL[]>(volume);

#pragma omp parallel shared(inptr, tmp)
    {
#pragma omp single
      merge_sort(inptr, 0, volume, &(tmp[0]));
    }

    if (is_index_space) {
      // not implemented yet
      assert(false);
    }

    if (!nested) omp_set_nested(0);
  }
};

/*static*/ void SortTask::omp_variant(TaskContext& context)
{
  sort_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
