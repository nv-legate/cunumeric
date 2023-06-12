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
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include <thrust/reduce.h>
#include <thrust/scatter.h>

#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <thrust/version.h>

#include <cub/cub.cuh>

#ifdef _DEBUG
#include <cassert>
#include <iostream>
#endif

#include "cunumeric/utilities/thrust_util.h"

#if THRUST_VERSION >= 101600
#pragma message "::par_nosync!"
#else
#pragma message "only ::par!"
#endif

namespace cunumeric {
namespace detail {
template <typename element_t>
decltype(auto) get_raw_ptr(thrust::device_vector<element_t>& v)
{
  return v.data().get();
}

template <typename element_t>
struct device_bin_generator_t {
  using vector_type = thrust::device_vector<element_t>;

  vector_type operator()(size_t num_bins, element_t min_bound, element_t max_bound) const
  {
    auto num_elems = num_bins + 1;
    vector_type d_v(num_elems);

    auto step = (max_bound - min_bound) / num_bins;

    thrust::transform(
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(num_elems),
      d_v.begin(),
      [step, min_bound] __device__(auto index) { return min_bound + index * step; });
    return d_v;
  }

  static element_t const* get_raw_ptr(vector_type const& d_v) { return d_v.data().get(); }
};

template <typename bin_t, typename range_t = bin_t, typename gen_t = device_bin_generator_t<bin_t>>
struct bin_manager_t {
  using vector_type = typename gen_t::vector_type;

  bin_manager_t(bin_t const* p_bins, size_t num_bins) : p_bins_(p_bins), num_bins_(num_bins) {}

  bin_manager_t(bin_t min_bound, bin_t max_bound, gen_t gen, size_t num_bins = 10)
    : num_bins_(num_bins), range_(std::make_pair(min_bound, max_bound))
  {
    bins_storage_ = gen(num_bins_, range_.first, range_.second);

    p_bins_ = gen_t::get_raw_ptr(bins_storage_);
  }

#ifdef _DEBUG
  // PROBLEM: returns empty container
  // when 1st constructor invoked;
  //
  vector_type generate_bins(int) const { return bins_storage_; }
#endif

  std::tuple<bin_t const*, size_t> generate_bins() const
  {
    return std::make_tuple(p_bins_, num_bins_);
  }

 private:
  bin_t const* p_bins_{nullptr};
  size_t num_bins_{10};
  std::pair<range_t, range_t> range_{0, 0};
  vector_type bins_storage_;
};

// rudimentary device allocator:
//
template <typename elem_t, typename exe_policy_t>
struct allocator_t {
  allocator_t(void) {}

  allocator_t(elem_t, exe_policy_t) {}  // tag-dispatch for CTAD

  elem_t* operator()(size_t n_bytes, elem_t init = 0)
  {
    d_buffer_ = thrust::device_vector<elem_t>(n_bytes, init);
    return get_raw_ptr(d_buffer_);
  }

 private:
  thrust::device_vector<elem_t> d_buffer_;
};

template <typename exe_policy_t>
void synchronize_exec(exe_policy_t)
{
  cudaDeviceSynchronize();
}

// device version:
//
template <typename exe_policy_t, typename weight_t, typename offset_t, typename allocator_t>
struct segmented_sum_t {
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
      ptr_offsets_(p_offsets),
      stream_(stream)
  {
    cub::DeviceSegmentedReduce::Sum(p_temp_storage_,
                                    temp_storage_bytes_,
                                    ptr_weights_,
                                    ptr_hist_,
                                    n_intervals_,
                                    ptr_offsets_,
                                    ptr_offsets_ + 1,
                                    stream_);

    p_temp_storage_ = allocator(temp_storage_bytes_);
  }

  void operator()(void)
  {
    cub::DeviceSegmentedReduce::Sum(p_temp_storage_,
                                    temp_storage_bytes_,
                                    ptr_weights_,
                                    ptr_hist_,
                                    n_intervals_,
                                    ptr_offsets_,
                                    ptr_offsets_ + 1,
                                    stream_);
  }

  weight_t* get_histogram(void) { return ptr_hist_; }

 private:
  weight_t const* ptr_weights_{nullptr};
  size_t n_samples_{0};
  weight_t* ptr_hist_{nullptr};
  size_t n_intervals_{0};
  offset_t* ptr_offsets_{nullptr};
  cudaStream_t stream_{nullptr};
  void* p_temp_storage_{nullptr};
  size_t temp_storage_bytes_{0};
};

}  // namespace detail
}  // namespace cunumeric
