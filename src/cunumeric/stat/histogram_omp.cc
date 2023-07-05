/* Copyright 2023 NVIDIA Corporation
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

#include "cunumeric/stat/histogram.h"
#include "cunumeric/stat/histogram_template.inl"

#include <omp.h>

#include <algorithm>
#include <numeric>
#include <tuple>

namespace cunumeric {
using namespace legate;

namespace detail {

// RO accessor (size, pointer) extractor:
//
template <typename VAL>
std::tuple<size_t, const VAL*> get_accessor_ptr(const AccessorRO<VAL, 1>& src_acc,
                                                const Rect<1>& src_rect)
{
  size_t src_strides[1];
  const VAL* src_ptr = src_acc.ptr(src_rect, src_strides);
  assert(src_strides[0] == 1);
  //
  // const VAL* src_ptr: need to create a copy with create_buffer(...);
  // since src will get sorted (in-place);
  //
  size_t src_size = src_rect.hi - src_rect.lo + 1;
  return std::make_tuple(src_size, src_ptr);
}
// RD accessor (size, pointer) extractor:
//
template <typename VAL>
std::tuple<size_t, VAL*> get_accessor_ptr(const AccessorRD<SumReduction<VAL>, true, 1>& src_acc,
                                          const Rect<1>& src_rect)
{
  size_t src_strides[1];
  VAL* src_ptr = src_acc.ptr(src_rect, src_strides);
  assert(src_strides[0] == 1);
  //
  // const VAL* src_ptr: need to create a copy with create_buffer(...);
  // since src will get sorted (in-place);
  //
  size_t src_size = src_rect.hi - src_rect.lo + 1;
  return std::make_tuple(src_size, src_ptr);
}
}  // namespace detail

template <Type::Code CODE>
struct HistogramImplBody<VariantKind::OMP, CODE> {
  using VAL = legate_type_of<CODE>;

  // for now, it has been decided to hardcode these types:
  //
  using BinType    = double;
  using WeightType = double;

  // in the future we might relax relax that requirement,
  // but complicate dispatching:
  //
  // template <typename BinType = VAL, typename WeightType = VAL>
  void operator()(const AccessorRO<VAL, 1>& src,
                  const Rect<1>& src_rect,
                  const AccessorRO<BinType, 1>& bins,
                  const Rect<1>& bins_rect,
                  const AccessorRO<WeightType, 1>& weights,
                  const Rect<1>& weights_rect,
                  const AccessorRD<SumReduction<WeightType>, true, 1>& result,
                  const Rect<1>& result_rect) const
  {
    auto&& [src_size, src_ptr] = detail::get_accessor_ptr(src, src_rect);

    auto&& [weights_size, weights_ptr] = detail::get_accessor_ptr(weights, weights_rect);

    auto&& [bins_size, bins_ptr] = detail::get_accessor_ptr(bins, bins_rect);

    auto num_intervals              = bins_size - 1;
    Buffer<WeightType> local_result = create_buffer<WeightType>(num_intervals);

    WeightType* local_result_ptr = local_result.ptr(0);

    auto&& [global_result_size, global_result_ptr] = detail::get_accessor_ptr(result, result_rect);

#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (auto bin_index = 0; bin_index < num_intervals; ++bin_index) {
        local_result_ptr[bin_index] = 0;

        if (bin_index == num_intervals - 1) {  // interior[l, r]
          for (auto src_index = 0; src_index < src_size; ++src_index) {
            if (src_ptr[src_index] >= bins_ptr[bin_index] &&
                src_ptr[src_index] <= bins_ptr[bin_index + 1]) {
              local_result_ptr[bin_index] += weights_ptr[src_index];
            }
          }
        } else {  // interior[l, r)
          for (auto src_index = 0; src_index < src_size; ++src_index) {
            if (src_ptr[src_index] >= bins_ptr[bin_index] &&
                src_ptr[src_index] < bins_ptr[bin_index + 1]) {
              local_result_ptr[bin_index] += weights_ptr[src_index];
            }
          }
        }
      }

      // #pragma omp barrier ?

      // fold into RD result:
      //
      assert(num_intervals == global_result_size);

#pragma omp for schedule(static)
      for (auto bin_index = 0; bin_index < num_intervals; ++bin_index) {
        global_result_ptr[bin_index] += local_result_ptr[bin_index];
      }
    }
  }
};

/*static*/ void HistogramTask::omp_variant(TaskContext& context)
{
  histogram_template<VariantKind::OMP>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  HistogramTask::register_variants();
}
}  // namespace

}  // namespace cunumeric
