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

#include "cunumeric/stat/histogram_gen.h"

namespace cunumeric {
namespace detail {

template <typename exe_policy_t, typename elem_t, typename bin_t>
struct lower_bound_op_t {
  lower_bound_op_t(exe_policy_t, bin_t const* p_bins, size_t n_intervs)
    : p_bins_(p_bins), n_intervs_(n_intervs)  // CTAD
  {
  }
  __host__ __device__ bool operator()(elem_t left, bin_t right) const
  {
    // sentinel logic accounts for comparison
    // against last bin's upper bound, when
    // (<) is to be replaced by (<=):
    //
    auto sentinel = p_bins_[n_intervs_];
    if constexpr (std::is_same_v<elem_t, __half> && (!std::is_integral_v<bin_t>)) {
      // upcast operands:
      //
      bin_t left_up  = static_cast<bin_t>(left);
      bin_t right_up = static_cast<bin_t>(right);
      if (left_up == sentinel && right_up == sentinel)
        return true;
      else
        return left_up < right_up;
    } else if constexpr (std::is_same_v<elem_t, __half> && std::is_integral_v<bin_t>) {
      // upcast sentinel:
      //
      elem_t sentinel_up = static_cast<elem_t>(sentinel);
      if (left == sentinel_up && right == sentinel_up)
        return true;
      else
        return left < right;
    } else {
      if (left == sentinel && right == sentinel)
        return true;
      else
        return left < right;
    }
  }

 private:
  bin_t const* p_bins_;
  size_t n_intervs_;
};

template <typename exe_policy_t, typename bin_t, typename weight_t>
struct transform_op_t {
  transform_op_t(exe_policy_t exe_pol, bin_t const* p_bins, weight_t* p_hist, size_t n_intervs)
    : ptr_bins_(p_bins)
  {
    Sw_ = thrust::reduce(exe_pol, p_hist, p_hist + n_intervs);
  }

  __host__ __device__ weight_t operator()(size_t index, weight_t h_i)
  {
    auto d_i = ptr_bins_[index + 1] - ptr_bins_[index];

#ifdef _ZERO_WIDTH_RET_ZERO_
    if (d_i == 0)
      return static_cast<weight_t>(0);
    else
#endif
      return h_i / (Sw_ * d_i);
  }

 private:
  bin_t const* ptr_bins_;
  weight_t* ptr_h_;
  weight_t Sw_{0};
};

template <typename exe_policy_t,
          typename elem_t,
          typename bin_t,
          template <typename...> typename alloc_t = allocator_t,
          typename weight_t                       = elem_t,
          typename offset_t                       = size_t>
void histogram_weights(exe_policy_t exe_pol,
                       elem_t* ptr_src,  // source array, a
                       size_t n_samples,
                       bin_t const* ptr_bins,          // bins array,
                       size_t n_intervals,             // |bins| - 1
                       weight_t* ptr_hist,             // result; pre-allocated, sz = n_intervals
                       weight_t* ptr_w     = nullptr,  // weights array, w
                       cudaStream_t stream = nullptr)
{
  alloc_t<offset_t, exe_policy_t> alloc_offsets;
  auto* ptr_offsets = alloc_offsets(exe_pol, n_intervals + 1);

  alloc_t<weight_t, exe_policy_t> alloc_w;

  if (!ptr_w) { ptr_w = alloc_w(exe_pol, n_samples, 1); }

  // in-place src sort + corresponding weights shuffle:
  //
  thrust::sort_by_key(exe_pol, ptr_src, ptr_src + n_samples, ptr_w);

  synchronize_exec(exe_pol, stream);

  // l-b functor:
  //
  lower_bound_op_t<exe_policy_t, elem_t, bin_t> lbop{exe_pol, ptr_bins, n_intervals};

  // vectorized lower-bounds of bins against src:
  //
  thrust::lower_bound(
    exe_pol, ptr_src, ptr_src + n_samples, ptr_bins, ptr_bins + n_intervals + 1, ptr_offsets, lbop);

  alloc_t<unsigned char, exe_policy_t> alloc_scratch;

  // needs explicit template args;
  // CTAD won't work with SFINAE;
  //
  segmented_sum_t<exe_policy_t, weight_t, offset_t, decltype(alloc_scratch)> segsum{
    exe_pol, ptr_w, n_samples, ptr_hist, n_intervals, ptr_offsets, stream, alloc_scratch};

  segsum();
}

}  // namespace detail
}  // namespace cunumeric
