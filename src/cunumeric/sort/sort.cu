/* Copyright 2021-2022 NVIDIA Corporation
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
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  copy_into_buffer(VAL* out,
                   const AccessorRO<VAL, DIM> accessor,
                   const Point<DIM> lo,
                   const Pitches<DIM - 1> pitches,
                   const size_t volume)
{
  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= volume) return;
  auto point  = pitches.unflatten(offset, lo);
  out[offset] = accessor[lo + point];
}

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  copy_into_output(AccessorWO<VAL, DIM> accessor,
                   const VAL* data,
                   const Point<DIM> lo,
                   const Pitches<DIM - 1> pitches,
                   const size_t volume)
{
  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= volume) return;
  auto point           = pitches.unflatten(offset, lo);
  accessor[lo + point] = data[offset];
}

struct multiply : public thrust::unary_function<int, int> {
  const int constant;

  multiply(int _constant) : constant(_constant) {}

  __host__ __device__ int operator()(int& input) const { return input * constant; }
};

template <class VAL>
void cub_local_sort_inplace(
  VAL* inptr, int64_t* argptr, const size_t volume, const size_t sort_dim_size, cudaStream_t stream)
{
  // make a copy of input --> we want inptr to return sorted values
  auto keys_in = create_buffer<VAL>(volume, Legion::Memory::Kind::GPU_FB_MEM);
  cudaMemcpyAsync(keys_in.ptr(0), inptr, sizeof(VAL) * volume, cudaMemcpyDeviceToDevice, stream);
  size_t temp_storage_bytes = 0;
  if (argptr == nullptr) {
    if (volume == sort_dim_size) {
      // sort
      cub::DeviceRadixSort::SortKeys(
        NULL, temp_storage_bytes, keys_in.ptr(0), inptr, volume, 0, sizeof(VAL) * 8, stream);
      auto temp_storage =
        create_buffer<unsigned char>(temp_storage_bytes, Legion::Memory::Kind::GPU_FB_MEM);
      cub::DeviceRadixSort::SortKeys(temp_storage.ptr(0),
                                     temp_storage_bytes,
                                     keys_in.ptr(0),
                                     inptr,
                                     volume,
                                     0,
                                     sizeof(VAL) * 8,
                                     stream);
    } else {
      // segmented sort
      auto off_start_it =
        thrust::make_transform_iterator(thrust::make_counting_iterator(0), multiply(sort_dim_size));
      auto off_end_it =
        thrust::make_transform_iterator(thrust::make_counting_iterator(1), multiply(sort_dim_size));

      cub::DeviceSegmentedRadixSort::SortKeys(NULL,
                                              temp_storage_bytes,
                                              keys_in.ptr(0),
                                              inptr,
                                              volume,
                                              volume / sort_dim_size,
                                              off_start_it,
                                              off_end_it,
                                              0,
                                              sizeof(VAL) * 8,
                                              stream);
      auto temp_storage =
        create_buffer<unsigned char>(temp_storage_bytes, Legion::Memory::Kind::GPU_FB_MEM);

      cub::DeviceSegmentedRadixSort::SortKeys(temp_storage.ptr(0),
                                              temp_storage_bytes,
                                              keys_in.ptr(0),
                                              inptr,
                                              volume,
                                              volume / sort_dim_size,
                                              off_start_it,
                                              off_end_it,
                                              0,
                                              sizeof(VAL) * 8,
                                              stream);
    }
  } else {
    auto idx_in = create_buffer<int64_t>(volume, Legion::Memory::Kind::GPU_FB_MEM);
    thrust::transform(thrust::cuda::par.on(stream),
                      thrust::make_counting_iterator<int64_t>(0),
                      thrust::make_counting_iterator<int64_t>(volume),
                      thrust::make_constant_iterator<int64_t>(sort_dim_size),
                      idx_in.ptr(0),
                      thrust::modulus<int64_t>());

    if (volume == sort_dim_size) {
      // argsort
      cub::DeviceRadixSort::SortPairs(NULL,
                                      temp_storage_bytes,
                                      keys_in.ptr(0),
                                      inptr,
                                      idx_in.ptr(0),
                                      argptr,
                                      volume,
                                      0,
                                      sizeof(VAL) * 8,
                                      stream);

      auto temp_storage =
        create_buffer<unsigned char>(temp_storage_bytes, Legion::Memory::Kind::GPU_FB_MEM);

      cub::DeviceRadixSort::SortPairs(temp_storage.ptr(0),
                                      temp_storage_bytes,
                                      keys_in.ptr(0),
                                      inptr,
                                      idx_in.ptr(0),
                                      argptr,
                                      volume,
                                      0,
                                      sizeof(VAL) * 8,
                                      stream);
    } else {
      // segmented argsort
      auto off_start_it =
        thrust::make_transform_iterator(thrust::make_counting_iterator(0), multiply(sort_dim_size));
      auto off_end_it =
        thrust::make_transform_iterator(thrust::make_counting_iterator(1), multiply(sort_dim_size));

      cub::DeviceSegmentedRadixSort::SortPairs(NULL,
                                               temp_storage_bytes,
                                               keys_in.ptr(0),
                                               inptr,
                                               idx_in.ptr(0),
                                               argptr,
                                               volume,
                                               volume / sort_dim_size,
                                               off_start_it,
                                               off_end_it,
                                               0,
                                               sizeof(VAL) * 8,
                                               stream);

      auto temp_storage =
        create_buffer<unsigned char>(temp_storage_bytes, Legion::Memory::Kind::GPU_FB_MEM);

      cub::DeviceSegmentedRadixSort::SortPairs(temp_storage.ptr(0),
                                               temp_storage_bytes,
                                               keys_in.ptr(0),
                                               inptr,
                                               idx_in.ptr(0),
                                               argptr,
                                               volume,
                                               volume / sort_dim_size,
                                               off_start_it,
                                               off_end_it,
                                               0,
                                               sizeof(VAL) * 8,
                                               stream);
    }
  }
}

template <class VAL>
void thrust_local_sort_inplace(
  VAL* inptr, int64_t* argptr, const size_t volume, const size_t sort_dim_size, cudaStream_t stream)
{
  if (argptr == nullptr) {
    if (volume == sort_dim_size) {
      thrust::stable_sort(thrust::cuda::par.on(stream), inptr, inptr + volume);
    } else {
      auto sort_id = create_buffer<uint64_t>(volume, Legion::Memory::Kind::GPU_FB_MEM);
      // init combined keys
      thrust::transform(thrust::cuda::par.on(stream),
                        thrust::make_counting_iterator<uint64_t>(0),
                        thrust::make_counting_iterator<uint64_t>(volume),
                        thrust::make_constant_iterator<uint64_t>(sort_dim_size),
                        sort_id.ptr(0),
                        thrust::divides<uint64_t>());
      auto combined = thrust::make_zip_iterator(thrust::make_tuple(sort_id.ptr(0), inptr));

      thrust::stable_sort(thrust::cuda::par.on(stream),
                          combined,
                          combined + volume,
                          thrust::less<thrust::tuple<size_t, VAL>>());
    }
  } else {
    // intialize indices
    thrust::transform(thrust::cuda::par.on(stream),
                      thrust::make_counting_iterator<int64_t>(0),
                      thrust::make_counting_iterator<int64_t>(volume),
                      thrust::make_constant_iterator<int64_t>(sort_dim_size),
                      argptr,
                      thrust::modulus<int64_t>());

    if (volume == sort_dim_size) {
      thrust::stable_sort_by_key(thrust::cuda::par.on(stream), inptr, inptr + volume, argptr);
    } else {
      auto sort_id = create_buffer<uint64_t>(volume, Legion::Memory::Kind::GPU_FB_MEM);
      // init combined keys
      thrust::transform(thrust::cuda::par.on(stream),
                        thrust::make_counting_iterator<uint64_t>(0),
                        thrust::make_counting_iterator<uint64_t>(volume),
                        thrust::make_constant_iterator<uint64_t>(sort_dim_size),
                        sort_id.ptr(0),
                        thrust::divides<uint64_t>());
      auto combined = thrust::make_zip_iterator(thrust::make_tuple(sort_id.ptr(0), inptr));

      thrust::stable_sort_by_key(thrust::cuda::par.on(stream),
                                 combined,
                                 combined + volume,
                                 argptr,
                                 thrust::less<thrust::tuple<size_t, VAL>>());
    }
  }
}

template <LegateTypeCode CODE>
struct support_cub : std::true_type {
};
template <>
struct support_cub<LegateTypeCode::COMPLEX64_LT> : std::false_type {
};
template <>
struct support_cub<LegateTypeCode::COMPLEX128_LT> : std::false_type {
};

template <LegateTypeCode CODE, std::enable_if_t<support_cub<CODE>::value>* = nullptr>
void local_sort_inplace(legate_type_of<CODE>* inptr,
                        int64_t* argptr,
                        const size_t volume,
                        const size_t sort_dim_size,
                        cudaStream_t stream)
{
  using VAL = legate_type_of<CODE>;
  cub_local_sort_inplace<VAL>(inptr, argptr, volume, sort_dim_size, stream);
}

template <LegateTypeCode CODE, std::enable_if_t<!support_cub<CODE>::value>* = nullptr>
void local_sort_inplace(legate_type_of<CODE>* inptr,
                        int64_t* argptr,
                        const size_t volume,
                        const size_t sort_dim_size,
                        cudaStream_t stream)
{
  using VAL = legate_type_of<CODE>;
  thrust_local_sort_inplace<VAL>(inptr, argptr, volume, sort_dim_size, stream);
}

template <LegateTypeCode CODE, int32_t DIM>
struct SortImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(const Array& input_array,
                  Array& output_array,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t volume,
                  const bool argsort,
                  const Legion::DomainPoint global_shape,
                  const bool is_index_space,
                  const Legion::DomainPoint index_point,
                  const Legion::Domain domain)
  {
    AccessorRO<VAL, DIM> input = input_array.read_accessor<VAL, DIM>(rect);

    bool dense = input.accessor.is_dense_row_major(rect);

#ifdef DEBUG_CUNUMERIC
    std::cout << "GPU(" << getRank(domain, index_point) << "): local size = " << volume
              << ", dist. = " << is_index_space << ", index_point = " << index_point
              << ", domain/volume = " << domain << "/" << domain.get_volume()
              << ", dense = " << dense << ", argsort. = " << argsort << std::endl;
#endif

    auto stream = get_cached_stream();

    const size_t sort_dim_size = global_shape[DIM - 1];
    assert(!is_index_space || DIM > 1);  // not implemented for now

    // make a copy of the input
    auto dense_input_copy = create_buffer<VAL>(volume, Legion::Memory::Kind::GPU_FB_MEM);
    if (dense) {
      cudaMemcpyAsync(dense_input_copy.ptr(0),
                      input.ptr(rect.lo),
                      sizeof(VAL) * volume,
                      cudaMemcpyDeviceToDevice,
                      stream);
    } else {
      const size_t num_blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      copy_into_buffer<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        dense_input_copy.ptr(0), input, rect.lo, pitches, volume);
    }

    // we need a buffer for argsort
    auto indices_buffer =
      create_buffer<int64_t>(argsort ? volume : 0, Legion::Memory::Kind::GPU_FB_MEM);

    // sort data
    local_sort_inplace<CODE>(dense_input_copy.ptr(0),
                             argsort ? indices_buffer.ptr(0) : nullptr,
                             volume,
                             sort_dim_size,
                             stream);

    // copy back data (we assume output partition to be aliged to input!)
    if (dense) {
      if (argsort) {
        AccessorWO<int64_t, DIM> output = output_array.write_accessor<int64_t, DIM>(rect);
        cudaMemcpyAsync(output.ptr(rect.lo),
                        indices_buffer.ptr(0),
                        sizeof(int64_t) * volume,
                        cudaMemcpyDeviceToDevice,
                        stream);
      } else {
        AccessorWO<VAL, DIM> output = output_array.write_accessor<VAL, DIM>(rect);
        cudaMemcpyAsync(output.ptr(rect.lo),
                        dense_input_copy.ptr(0),
                        sizeof(VAL) * volume,
                        cudaMemcpyDeviceToDevice,
                        stream);
      }
    } else {
      const size_t num_blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      if (argsort) {
        AccessorWO<int64_t, DIM> output = output_array.write_accessor<int64_t, DIM>(rect);
        copy_into_output<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
          output, indices_buffer.ptr(0), rect.lo, pitches, volume);
      } else {
        AccessorWO<VAL, DIM> output = output_array.write_accessor<VAL, DIM>(rect);
        copy_into_output<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
          output, dense_input_copy.ptr(0), rect.lo, pitches, volume);
      }
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }
};

/*static*/ void SortTask::gpu_variant(TaskContext& context)
{
  sort_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
