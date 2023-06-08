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

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <thrust/binary_search.h>
#include <thrust/sort.h>

#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <thrust/version.h>

#include <cub/cub.cuh>

namespace cunumeric {
namespace detail {

template <typename element_t, template <typename...> typename vect_t>
decltype(auto) get_raw_ptr(vect_t<element_t>& v)
{
  if constexpr (std::is_same_v<vect_t<element_t>, thrust::device_vector<element_t>>) {
    return v.data().get();
  } else {
    return &v[0];
  }
}

template <typename exe_policy_t>
void synchronize_exec(exe_policy_t)
{
  cudaDeviceSynchronize();
}

template <>
void synchronize_exec(thrust::detail::host_t)
{
  // nothing;
}

template <typename array_t, typename bin_t>
struct lower_bound_op_t {
  lower_bound_op_t(bin_t const* p_bins, size_t n_intervs) : p_bins_(p_bins), n_intervs_(n_intervs)
  {
  }
  __host__ __device__ bool operator()(array_t left, array_t right) const
  {
    // sentinel logic accounts for comparison
    // against last bin's upper bound, when
    // (<) is to be replaced by (<=):
    //
    auto sentinel = p_bins_[n_intervs_];
    if (left == sentinel && right == sentinel)
      return true;
    else
      return left < right;
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
    if (d_i == 0)
      return static_cast<weight_t>(0);
    else
      return h_i / (Sw_ * d_i);
  }

 private:
  bin_t const* ptr_bins_;
  weight_t* ptr_h_;
  weight_t Sw_{0};
};

// rudimentary device allocator:
//
template <typename elem_t, typename exe_policy_t>
struct allocator_t {
  allocator_t(elem_t, exe_policy_t) {}  // tag-dispatch for CTAD

  elem_t* operator()(size_t n_bytes)
  {
    d_buffer_.resize(n_bytes);
    return get_raw_ptr(d_buffer_);
  }

 private:
  thrust::device_vector<elem_t> d_buffer_;
};

// host specialization
//
template <typename elem_t>
struct allocator_t<elem_t, thrust::detail::host_t> {
  allocator_t(elem_t, thrust::detail::host_t) {}  // tag-dispatch for CTAD

  elem_t* operator()(size_t n_bytes) const
  {
    h_buffer_.resize(n_bytes);
    return get_raw_ptr(h_buffer_);
  }

 private:
  std::vector<elem_t> h_buffer_;
};

// segmented-sum device version:
//
template <typename exe_policy_t,
          typename weight_t,
          typename offset_t,
          // template <typename...>
          typename allocator_t>
struct segmented_sum_t {
  segmented_sum_t(exe_policy_t exe_pol,
                  weight_t const* p_weights,
                  weight_t* p_hist,
                  size_t n_intervals,
                  offset_t* p_offsets,
                  cudaStream_t stream,
                  allocator_t& allocator)
    : ptr_weights_(p_weights),
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

  weight_t* get_histogram(void) { return ptr_weights_; }

 private:
  weight_t const* ptr_weights_{nullptr};
  weight_t* ptr_hist_{nullptr};
  size_t n_intervals_{0};
  offset_t* ptr_offsets_{nullptr};
  cudaStream_t stream_{nullptr};
  void* p_temp_storage_{nullptr};
  size_t temp_storage_bytes_{0};
};

// segmented-sum host specialization:
//
template <typename weight_t,
          typename offset_t,
          // template <typename...>
          typename allocator_t>
struct segmented_sum_t<thrust::detail::host_t, weight_t, offset_t, allocator_t> {
  // TODO:
};

// generic histogram algorithm
// targetting GPU and CPU
//
template <typename exe_policy_t,
          typename array_t,
          typename bin_t,
          typename alloc_t,
          template <typename...> typename vect_t = thrust::device_vector,
          typename weight_t                      = array_t,
          typename offset_t                      = size_t>
void histogram_weights(exe_policy_t exe_pol,
                       array_t* ptr_src,       // source array, a
                       size_t n_samples,       // size, |a|
                       bin_t const* ptr_over,  // bins array,
                       size_t n_intervals,     // |bins| - 1
                       weight_t* ptr_hist,     // result; pre-allocated, sz = n_intervals
                       alloc_t& allocator,
                       weight_t* ptr_w     = nullptr,  // weights array, w
                       bool density        = false,    // normalization
                       cudaStream_t stream = nullptr)
{
  vect_t<offset_t> v_offsets(n_intervals + 1);
  vect_t<weight_t> v_w;

  if (!ptr_w) {
    v_w   = vect_t<weight_t>(n_samples, 1);
    ptr_w = get_raw_ptr(v_w);
  }

  // in-place src sort + corresponding weights shuffle:
  //
  thrust::sort_by_key(exe_pol, ptr_src, ptr_src + n_samples, ptr_w);

  // synchronize_exec(exe_pol);

  // l-b functor:
  //
  lower_bound_op_t<array_t, bin_t> lbop{ptr_over, n_intervals};

  // vectorized lower-bounds of bins against src:
  //
  thrust::lower_bound(exe_pol,
                      ptr_src,
                      ptr_src + n_samples,
                      ptr_over,
                      ptr_over + n_intervals + 1,
                      get_raw_ptr(v_offsets),
                      lbop);

  segmented_sum_t segsum{
    exe_pol, ptr_w, ptr_hist, n_intervals, get_raw_ptr(v_offsets), stream, allocator};

  segsum();

  if (density) {
    transform_op_t top{exe_pol, ptr_over, ptr_hist, n_intervals};

    thrust::transform(exe_pol,
                      thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator<size_t>(n_intervals),
                      ptr_hist,
                      ptr_hist,  // in-place
                      top);
  }
}
}  // namespace detail
}  // namespace cunumeric
