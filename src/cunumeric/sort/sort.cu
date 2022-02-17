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

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;

template <LegateTypeCode CODE, int32_t DIM>
struct SortImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void thrust_sort(const VAL* inptr, VAL* outptr, const size_t volume, const size_t sort_dim_size)
  {
    thrust::device_ptr<const VAL> dev_input_ptr(inptr);
    thrust::device_ptr<VAL> dev_output_ptr(outptr);
    thrust::copy(dev_input_ptr, dev_input_ptr + volume, dev_output_ptr);
    // same approach as cupy implemntation --> combine multiple individual sorts into single
    // kernel with data tuples - (id_sub-sort, actual_data)
    if (DIM == 1) {
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

        for (uint64_t sort_part = 0; sort_part < number_sorts;
             sort_part += number_sorts_per_kernel) {
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

  void operator()(AccessorRO<VAL, DIM> input,
                  AccessorWO<VAL, DIM> output,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const bool dense,
                  const size_t volume,
                  const Legion::DomainPoint global_shape,
                  const bool is_index_space,
                  const Legion::DomainPoint index_point,
                  const Legion::Domain domain)
  {
#ifdef DEBUG_CUNUMERIC
    std::cout << "GPU(" << index_point[0] << "): local size = " << volume
              << ", dist. = " << is_index_space << ", index_point = " << index_point
              << ", domain/volume = " << domain << "/" << domain.get_volume()
              << ", dense = " << dense << std::endl;
#endif
    const size_t sort_dim_size = global_shape[DIM - 1];
    assert(!is_index_space || DIM > 1);  // not implemented for now
    if (dense) {
      thrust_sort(input.ptr(rect), output.ptr(rect), volume, sort_dim_size);
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
        thrust_sort(
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
