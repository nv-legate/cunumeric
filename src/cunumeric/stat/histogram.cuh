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
#include <thrust/fill.h>
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

// TODO: remove:
//
#if THRUST_VERSION >= 101600
#pragma message "::par_nosync!"
#else
#pragma message "only ::par!"
#endif

namespace cunumeric {
namespace detail {
template <typename element_t>
decltype(auto) get_raw_ptr(Buffer<element_t>& v)
{
  return v.ptr(0);
}

// rudimentary device allocator:
//
template <typename elem_t, typename exe_policy_t>
struct allocator_t {
  allocator_t(void) {}

  allocator_t(elem_t, exe_policy_t) {}  // tag-dispatch for CTAD

  elem_t* operator()(exe_policy_t exe_pol, size_t size, elem_t init = 0)
  {
    d_buffer_     = create_buffer<elem_t>(size, Legion::Memory::Kind::GPU_FB_MEM);
    elem_t* d_ptr = get_raw_ptr(d_buffer_);

    thrust::fill_n(exe_pol, d_ptr, size, init);

    return d_ptr;
  }

 private:
  Buffer<elem_t> d_buffer_;
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

    p_temp_storage_ = allocator(exe_pol, temp_storage_bytes_);
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