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

#include "cunumeric/cuda_help.h"

#include "cunumeric/stat/histogram.cuh"
#include "cunumeric/stat/histogram_impl.h"

#include "cunumeric/utilities/thrust_util.h"

#include <tuple>

namespace cunumeric {
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
// accessor copy utility:
//
template <typename VAL>
std::tuple<size_t, Buffer<VAL>, const VAL*> make_accessor_copy(const AccessorRO<VAL, 1>& src_acc,
                                                               const Rect<1>& src_rect)
{
  size_t src_strides[1];
  const VAL* src_ptr = src_acc.ptr(src_rect, src_strides);
  assert(src_strides[0] == 1);
  //
  // const VAL* src_ptr: need to create a copy with create_buffer(...);
  // since src will get sorted (in-place);
  //
  size_t src_size      = src_rect.hi - src_rect.lo + 1;
  Buffer<VAL> src_copy = create_buffer<VAL>(src_size, Legion::Memory::Kind::GPU_FB_MEM);
  return std::make_tuple(src_size, src_copy, src_ptr);
}
}  // namespace detail

template <Type::Code CODE>
struct HistogramImplBody<VariantKind::GPU, CODE> {
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
    auto stream          = get_cached_stream();
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);

    auto&& [src_size, src_copy, src_ptr] = detail::make_accessor_copy(src, src_rect);
    CHECK_CUDA(
      cudaMemcpyAsync(src_copy.ptr(0), src_ptr, src_size, cudaMemcpyDeviceToDevice, stream));

    auto&& [weights_size, weights_copy, weights_ptr] =
      detail::make_accessor_copy(weights, weights_rect);
    CHECK_CUDA(cudaMemcpyAsync(
      weights_copy.ptr(0), weights_ptr, weights_size, cudaMemcpyDeviceToDevice, stream));

    auto&& [bins_size, bins_ptr] = detail::get_accessor_ptr(bins, bins_rect);

    auto num_intervals = bins_size - 1;
    Buffer<WeightType> local_result =
      create_buffer<WeightType>(num_intervals, Legion::Memory::Kind::GPU_FB_MEM);

    WeightType* local_result_ptr = local_result.ptr(0);

    auto&& [global_result_size, global_result_ptr] = detail::get_accessor_ptr(result, result_rect);

    CHECK_CUDA(cudaStreamSynchronize(stream));

#if 0
    {
      // VAL* src_ptr                 = nullptr;
      // size_t src_size              = 0;
      // WeightType* bins_ptr         = nullptr;
      // size_t num_intervals         = 0;
      // WeightType* local_result_ptr = nullptr;
      // WeightType* weights_ptr = nullptr;

      detail::histogram_proto(DEFAULT_POLICY.on(stream),
                              src_copy.ptr(0),  // src_ptr compiled...why?
                              src_size,
                              bins_ptr,
                              num_intervals,
                              local_result_ptr,
                              weights_copy.ptr(0),  // weights_ptr,  // <- PROBLEM here
                              stream_);
    }
#endif

    // #if 0
    detail::histogram_weights(DEFAULT_POLICY.on(stream),
                              src_copy.ptr(0),
                              src_size,
                              bins_ptr,
                              num_intervals,
                              local_result_ptr,
                              weights_copy.ptr(0),
                              stream_);
    // #endif

    CHECK_CUDA(cudaStreamSynchronize(stream));

    // fold into RD result:
    //
    assert(num_intervals == global_result_size);

    thrust::transform(
      DEFAULT_POLICY.on(stream),
      local_result_ptr,
      local_result_ptr + num_intervals,
      global_result_ptr,
      global_result_ptr,
      [] __device__(auto local_value, auto global_value) { return local_value + global_value; });

    CHECK_CUDA(cudaStreamSynchronize(stream));
  }
};

/*static*/ void HistogramTask::gpu_variant(TaskContext& context)
{
  histogram_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
