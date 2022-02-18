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

struct multiply : public thrust::unary_function<int, int> {
  const int constant;

  multiply(int _constant) : constant(_constant) {}

  __host__ __device__ int operator()(int& input) const { return input * constant; }
};

template <class VAL>
void cub_sort(const VAL* inptr, VAL* outptr, const size_t volume, const size_t sort_dim_size)
{
  if (volume == sort_dim_size) {
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(NULL, temp_storage_bytes, inptr, outptr, volume);

    auto temp_storage =
      create_buffer<unsigned char>(temp_storage_bytes, Legion::Memory::Kind::GPU_FB_MEM);

    cub::DeviceRadixSort::SortKeys(temp_storage.ptr(0), temp_storage_bytes, inptr, outptr, volume);
  } else {
    auto off_start_it =
      thrust::make_transform_iterator(thrust::make_counting_iterator(0), multiply(sort_dim_size));
    auto off_end_it =
      thrust::make_transform_iterator(thrust::make_counting_iterator(1), multiply(sort_dim_size));

    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortKeys(NULL,
                                            temp_storage_bytes,
                                            inptr,
                                            outptr,
                                            volume,
                                            volume / sort_dim_size,
                                            off_start_it,
                                            off_end_it);
    auto temp_storage =
      create_buffer<unsigned char>(temp_storage_bytes, Legion::Memory::Kind::GPU_FB_MEM);

    cub::DeviceSegmentedRadixSort::SortKeys(temp_storage.ptr(0),
                                            temp_storage_bytes,
                                            inptr,
                                            outptr,
                                            volume,
                                            volume / sort_dim_size,
                                            off_start_it,
                                            off_end_it);
  }
}

template <class VAL>
void thrust_sort(const VAL* inptr, VAL* outptr, const size_t volume, const size_t sort_dim_size)
{
  thrust::device_ptr<const VAL> dev_input_ptr(inptr);
  thrust::device_ptr<VAL> dev_output_ptr(outptr);
  thrust::copy(dev_input_ptr, dev_input_ptr + volume, dev_output_ptr);
  // same approach as cupy implemntation --> combine multiple individual sorts into single
  // kernel with data tuples - (id_sub-sort, actual_data)
  if (volume == sort_dim_size) {
    thrust::stable_sort(dev_output_ptr, dev_output_ptr + volume);
  } else {
    // in this case we know we are sorting for the *last* index
    const uint64_t max_elements_per_kernel =
      1 << 22;  // TODO check amount of available GPU memory from config
    const uint64_t number_sorts_per_kernel =
      std::max(1ul, std::min(volume, max_elements_per_kernel) / sort_dim_size);
    const uint64_t number_sorts = volume / sort_dim_size;

    // std::cout << "Number of sorts per kernel: " << number_sorts_per_kernel << std::endl;

    if (number_sorts_per_kernel >=
        32)  // key-tuple sort has quite some overhead -- only utilize if beneficial
    {
      // allocate memory for keys (iterating +=1 for each individual sort dimension)
      // ensure keys have minimal bit-length (needs values up to number_sorts_per_kernel-1)!
      // TODO!!!!
      auto keys_array = create_buffer<uint32_t>(number_sorts_per_kernel * sort_dim_size,
                                                Legion::Memory::Kind::GPU_FB_MEM);
      thrust::device_ptr<uint32_t> dev_key_ptr(keys_array.ptr(0));

      for (uint64_t sort_part = 0; sort_part < number_sorts; sort_part += number_sorts_per_kernel) {
        // compute size of batch (might be smaller for the last call)
        const uint64_t num_elements =
          std::min(number_sorts - sort_part, max_elements_per_kernel) * sort_dim_size;
        const uint64_t offset = sort_part * sort_dim_size;

        // reinit keys
        thrust::transform(thrust::make_counting_iterator<uint64_t>(0),
                          thrust::make_counting_iterator<uint64_t>(num_elements),
                          thrust::make_constant_iterator<uint64_t>(sort_dim_size),
                          dev_key_ptr,
                          thrust::divides<uint64_t>());

        // sort
        auto combined =
          thrust::make_zip_iterator(thrust::make_tuple(dev_key_ptr, dev_output_ptr + offset));
        thrust::stable_sort(
          combined, combined + num_elements, thrust::less<thrust::tuple<size_t, VAL>>());
      }
    } else {
      // number_sorts_per_kernel too small ----> we sort one after another
      for (uint64_t sort_part = 0; sort_part < number_sorts; sort_part++) {
        const uint64_t offset = sort_part * sort_dim_size;
        thrust::stable_sort(dev_output_ptr + offset, dev_output_ptr + offset + sort_dim_size);
      }
    }
  }
}

template <class VAL>
void cub_argsort(const VAL* inptr, int32_t* outptr, const size_t volume, const size_t sort_dim_size)
{
  auto keys_out = create_buffer<VAL>(volume, Legion::Memory::Kind::GPU_FB_MEM);
  thrust::device_ptr<VAL> dev_key_out_ptr(keys_out.ptr(0));

  auto idx_in = create_buffer<int32_t>(volume, Legion::Memory::Kind::GPU_FB_MEM);
  thrust::device_ptr<int32_t> dev_idx_in_ptr(idx_in.ptr(0));
  thrust::transform(thrust::make_counting_iterator<int32_t>(0),
                    thrust::make_counting_iterator<int32_t>(volume),
                    thrust::make_constant_iterator<int32_t>(sort_dim_size),
                    dev_idx_in_ptr,
                    thrust::modulus<int32_t>());

  if (volume == sort_dim_size) {
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes, inptr, keys_out.ptr(0), idx_in.ptr(0), outptr, volume);

    auto temp_storage =
      create_buffer<unsigned char>(temp_storage_bytes, Legion::Memory::Kind::GPU_FB_MEM);

    cub::DeviceRadixSort::SortPairs(temp_storage.ptr(0),
                                    temp_storage_bytes,
                                    inptr,
                                    keys_out.ptr(0),
                                    idx_in.ptr(0),
                                    outptr,
                                    volume);
  } else {
    auto off_start_it =
      thrust::make_transform_iterator(thrust::make_counting_iterator(0), multiply(sort_dim_size));
    auto off_end_it =
      thrust::make_transform_iterator(thrust::make_counting_iterator(1), multiply(sort_dim_size));

    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(NULL,
                                             temp_storage_bytes,
                                             inptr,
                                             keys_out.ptr(0),
                                             idx_in.ptr(0),
                                             outptr,
                                             volume,
                                             volume / sort_dim_size,
                                             off_start_it,
                                             off_end_it);

    auto temp_storage =
      create_buffer<unsigned char>(temp_storage_bytes, Legion::Memory::Kind::GPU_FB_MEM);

    cub::DeviceSegmentedRadixSort::SortPairs(temp_storage.ptr(0),
                                             temp_storage_bytes,
                                             inptr,
                                             keys_out.ptr(0),
                                             idx_in.ptr(0),
                                             outptr,
                                             volume,
                                             volume / sort_dim_size,
                                             off_start_it,
                                             off_end_it);
  }
}

template <class VAL>
void thrust_argsort(const VAL* inptr,
                    int32_t* outptr,
                    const size_t volume,
                    const size_t sort_dim_size)
{
  thrust::device_ptr<const VAL> dev_input_ptr(inptr);

  auto keys_copy = create_buffer<VAL>(volume, Legion::Memory::Kind::GPU_FB_MEM);
  thrust::device_ptr<VAL> dev_keys_copy_ptr(keys_copy.ptr(0));
  thrust::copy(dev_input_ptr, dev_input_ptr + volume, dev_keys_copy_ptr);

  thrust::device_ptr<int32_t> dev_output_ptr(outptr);
  thrust::transform(thrust::make_counting_iterator<int32_t>(0),
                    thrust::make_counting_iterator<int32_t>(volume),
                    thrust::make_constant_iterator<int32_t>(sort_dim_size),
                    dev_output_ptr,
                    thrust::modulus<int32_t>());

  // same approach as cupy implemntation --> combine multiple individual sorts into single
  // kernel with data tuples - (id_sub-sort, actual_data)
  if (volume == sort_dim_size) {
    thrust::stable_sort_by_key(dev_keys_copy_ptr, dev_keys_copy_ptr + volume, dev_output_ptr);
  } else {
    // in this case we know we are sorting for the *last* index
    const uint64_t max_elements_per_kernel =
      1 << 22;  // TODO check amount of available GPU memory from config
    const uint64_t number_sorts_per_kernel =
      std::max(1ul, std::min(volume, max_elements_per_kernel) / sort_dim_size);
    const uint64_t number_sorts = volume / sort_dim_size;

    // std::cout << "Number of sorts per kernel: " << number_sorts_per_kernel << std::endl;

    if (number_sorts_per_kernel >=
        32)  // key-tuple sort has quite some overhead -- only utilize if beneficial
    {
      // allocate memory for keys (iterating +=1 for each individual sort dimension)
      // ensure keys have minimal bit-length (needs values up to number_sorts_per_kernel-1)!
      // TODO!!!!
      auto keys_array = create_buffer<uint32_t>(number_sorts_per_kernel * sort_dim_size,
                                                Legion::Memory::Kind::GPU_FB_MEM);
      thrust::device_ptr<uint32_t> dev_key_ptr(keys_array.ptr(0));

      for (uint64_t sort_part = 0; sort_part < number_sorts; sort_part += number_sorts_per_kernel) {
        // compute size of batch (might be smaller for the last call)
        const uint64_t num_elements =
          std::min(number_sorts - sort_part, max_elements_per_kernel) * sort_dim_size;
        const uint64_t offset = sort_part * sort_dim_size;

        // reinit keys
        thrust::transform(thrust::make_counting_iterator<uint64_t>(0),
                          thrust::make_counting_iterator<uint64_t>(num_elements),
                          thrust::make_constant_iterator<uint64_t>(sort_dim_size),
                          dev_key_ptr,
                          thrust::divides<uint64_t>());

        // sort
        auto combined =
          thrust::make_zip_iterator(thrust::make_tuple(dev_key_ptr, dev_keys_copy_ptr + offset));
        thrust::stable_sort_by_key(combined,
                                   combined + num_elements,
                                   dev_output_ptr + offset,
                                   thrust::less<thrust::tuple<size_t, VAL>>());
      }
    } else {
      // number_sorts_per_kernel too small ----> we sort one after another
      for (uint64_t sort_part = 0; sort_part < number_sorts; sort_part++) {
        const uint64_t offset = sort_part * sort_dim_size;
        thrust::stable_sort_by_key(dev_keys_copy_ptr + offset,
                                   dev_keys_copy_ptr + offset + sort_dim_size,
                                   dev_output_ptr + offset);
      }
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
void sort_stable(const legate_type_of<CODE>* inptr,
                 legate_type_of<CODE>* outptr,
                 const size_t volume,
                 const size_t sort_dim_size)
{
  using VAL = legate_type_of<CODE>;
  cub_sort<VAL>(inptr, outptr, volume, sort_dim_size);
}

template <LegateTypeCode CODE, std::enable_if_t<!support_cub<CODE>::value>* = nullptr>
void sort_stable(const legate_type_of<CODE>* inptr,
                 legate_type_of<CODE>* outptr,
                 const size_t volume,
                 const size_t sort_dim_size)
{
  using VAL = legate_type_of<CODE>;
  thrust_sort<VAL>(inptr, outptr, volume, sort_dim_size);
}

template <LegateTypeCode CODE, std::enable_if_t<support_cub<CODE>::value>* = nullptr>
void argsort_stable(const legate_type_of<CODE>* inptr,
                    int32_t* outptr,
                    const size_t volume,
                    const size_t sort_dim_size)
{
  using VAL = legate_type_of<CODE>;
  cub_argsort<VAL>(inptr, outptr, volume, sort_dim_size);
}

template <LegateTypeCode CODE, std::enable_if_t<!support_cub<CODE>::value>* = nullptr>
void argsort_stable(const legate_type_of<CODE>* inptr,
                    int32_t* outptr,
                    const size_t volume,
                    const size_t sort_dim_size)
{
  using VAL = legate_type_of<CODE>;
  thrust_argsort<VAL>(inptr, outptr, volume, sort_dim_size);
}

template <LegateTypeCode CODE, int32_t DIM>
struct SortImplBody<VariantKind::GPU, false, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(AccessorRO<VAL, DIM> input,
                  AccessorWO<VAL, DIM> output,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const bool dense,
                  const size_t volume,
                  const bool argsort,
                  const Legion::DomainPoint global_shape,
                  const bool is_index_space,
                  const Legion::DomainPoint index_point,
                  const Legion::Domain domain)
  {
#ifdef DEBUG_CUNUMERIC
    std::cout << "GPU(" << getRank(domain, index_point) << "): local size = " << volume
              << ", dist. = " << is_index_space << ", index_point = " << index_point
              << ", domain/volume = " << domain << "/" << domain.get_volume()
              << ", dense = " << dense << std::endl;
#endif
    assert(!argsort);
    const size_t sort_dim_size = global_shape[DIM - 1];
    assert(!is_index_space || DIM > 1);  // not implemented for now
    if (dense) {
      sort_stable<CODE>(input.ptr(rect), output.ptr(rect), volume, sort_dim_size);
    } else {
      // compute contiguous memory block
      int contiguous_elements = 1;
      for (int i = DIM - 1; i >= 0; i--) {
        auto diff = 1 + rect.hi[i] - rect.lo[i];
        contiguous_elements *= diff;
        if (diff < global_shape[i]) { break; }
      }

      uint64_t elements_processed = 0;
      while (elements_processed < volume) {
        Legion::Point<DIM> start_point = pitches.unflatten(elements_processed, rect.lo);
        sort_stable<CODE>(
          input.ptr(start_point), output.ptr(start_point), contiguous_elements, sort_dim_size);
        elements_processed += contiguous_elements;
      }
    }
  }
};

template <LegateTypeCode CODE, int32_t DIM>
struct SortImplBody<VariantKind::GPU, true, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(AccessorRO<VAL, DIM> input,
                  AccessorWO<int32_t, DIM> output,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const bool dense,
                  const size_t volume,
                  const bool argsort,
                  const Legion::DomainPoint global_shape,
                  const bool is_index_space,
                  const Legion::DomainPoint index_point,
                  const Legion::Domain domain)
  {
#ifdef DEBUG_CUNUMERIC
    std::cout << "GPU(" << getRank(domain, index_point) << "): local size = " << volume
              << ", dist. = " << is_index_space << ", index_point = " << index_point
              << ", domain/volume = " << domain << "/" << domain.get_volume()
              << ", dense = " << dense << std::endl;
#endif
    assert(argsort);
    const size_t sort_dim_size = global_shape[DIM - 1];
    assert(!is_index_space || DIM > 1);  // not implemented for now
    if (dense) {
      argsort_stable<CODE>(input.ptr(rect), output.ptr(rect), volume, sort_dim_size);
    } else {
      // compute contiguous memory block
      int contiguous_elements = 1;
      for (int i = DIM - 1; i >= 0; i--) {
        auto diff = 1 + rect.hi[i] - rect.lo[i];
        contiguous_elements *= diff;
        if (diff < global_shape[i]) { break; }
      }

      uint64_t elements_processed = 0;
      while (elements_processed < volume) {
        Legion::Point<DIM> start_point = pitches.unflatten(elements_processed, rect.lo);
        argsort_stable<CODE>(
          input.ptr(start_point), output.ptr(start_point), contiguous_elements, sort_dim_size);
        elements_processed += contiguous_elements;
      }
    }
  }
};

/*static*/ void SortTask::gpu_variant(TaskContext& context)
{
  sort_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
