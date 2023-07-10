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

#pragma once

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include "cunumeric/stat/histogram_gen.h"

namespace cunumeric {
namespace detail {
template <typename element_t>
decltype(auto) get_raw_ptr(Buffer<element_t>& v)
{
  return v.ptr(0);
}

// host specialization
//
template <typename elem_t, typename exe_policy_t>
struct allocator_t<
  elem_t,
  exe_policy_t,
  std::enable_if_t<std::is_same_v<exe_policy_t, thrust::detail::host_t> ||
                   std::is_same_v<exe_policy_t, thrust::system::omp::detail::par_t>>> {
  allocator_t(void) {}

  allocator_t(elem_t, exe_policy_t) {}  // tag-dispatch for CTAD

  elem_t* operator()(exe_policy_t exe_pol, size_t size, elem_t init = 0)
  {
    d_buffer_     = create_buffer<elem_t>(size);
    elem_t* d_ptr = get_raw_ptr(d_buffer_);

    std::fill_n(d_ptr, size, init);

    return d_ptr;
  }

 private:
  Buffer<elem_t> d_buffer_;
};

template <typename exe_policy_t>
void synchronize_exec(exe_policy_t, cudaStream_t stream)
{
  // nothing;
}

// host specialization:
//
template <typename exe_policy_t, typename weight_t, typename offset_t, typename allocator_t>
struct segmented_sum_t<
  exe_policy_t,
  weight_t,
  offset_t,
  allocator_t,
  std::enable_if_t<std::is_same_v<exe_policy_t, thrust::detail::host_t> ||
                   std::is_same_v<exe_policy_t, thrust::system::omp::detail::par_t>>> {
  segmented_sum_t(exe_policy_t exe_pol,
                  weight_t const* p_weights,
                  size_t n_samples,
                  weight_t* p_hist,
                  size_t n_intervals,
                  offset_t* p_offsets,
                  cudaStream_t stream,
                  allocator_t& allocator)
    : ptr_weights_(p_weights),
      n_samples_(n_samples),
      ptr_hist_(p_hist),
      n_intervals_(n_intervals),
      ptr_offsets_(p_offsets)
  {
  }
  void operator()(void)
  {
    for (auto interval_index = 0; interval_index < n_intervals_; ++interval_index) {
      auto offset_index = interval_index + 1;
      auto* it_right    = ptr_weights_ + ptr_offsets_[offset_index];

      // if (ptr_offsets_[offset_index] == ptr_offsets_[n_intervals_])
      //  equivalent:
      //
      if (offset_index == n_intervals_) it_right = ptr_weights_ + n_samples_;

      ptr_hist_[interval_index] =
        std::accumulate(ptr_weights_ + ptr_offsets_[interval_index], it_right, 0.0);
    }
  }

  weight_t* get_histogram(void) { return ptr_hist_; }

 private:
  weight_t const* ptr_weights_{nullptr};
  size_t n_samples_{0};
  weight_t* ptr_hist_{nullptr};
  size_t n_intervals_{0};
  offset_t* ptr_offsets_{nullptr};
};

}  // namespace detail
}  // namespace cunumeric
