/* Copyright 2022 NVIDIA Corporation
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
#include <thrust/binary_search.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/thread/thread_search.cuh>

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
  CHECK_CUDA(
    cudaMemcpyAsync(keys_in.ptr(0), inptr, sizeof(VAL) * volume, cudaMemcpyDeviceToDevice, stream));
  size_t temp_storage_bytes = 0;
  if (argptr == nullptr) {
    if (volume == sort_dim_size) {
      // sort (initial call to compute bufffer size)
      cub::DeviceRadixSort::SortKeys(
        nullptr, temp_storage_bytes, keys_in.ptr(0), inptr, volume, 0, sizeof(VAL) * 8, stream);
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
      // segmented sort (initial call to compute bufffer size)
      auto off_start_it =
        thrust::make_transform_iterator(thrust::make_counting_iterator(0), multiply(sort_dim_size));
      auto off_end_it =
        thrust::make_transform_iterator(thrust::make_counting_iterator(1), multiply(sort_dim_size));

      cub::DeviceSegmentedRadixSort::SortKeys(nullptr,
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
    CHECK_CUDA(cudaMemcpyAsync(
      idx_in.ptr(0), argptr, sizeof(int64_t) * volume, cudaMemcpyDeviceToDevice, stream));

    if (volume == sort_dim_size) {
      // argsort (initial call to compute bufffer size)
      cub::DeviceRadixSort::SortPairs(nullptr,
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
      // segmented argsort (initial call to compute bufffer size)
      auto off_start_it =
        thrust::make_transform_iterator(thrust::make_counting_iterator(0), multiply(sort_dim_size));
      auto off_end_it =
        thrust::make_transform_iterator(thrust::make_counting_iterator(1), multiply(sort_dim_size));

      cub::DeviceSegmentedRadixSort::SortPairs(nullptr,
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
      thrust::sort(thrust::cuda::par.on(stream), inptr, inptr + volume);
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

      thrust::sort(thrust::cuda::par.on(stream),
                   combined,
                   combined + volume,
                   thrust::less<thrust::tuple<size_t, VAL>>());
    }
  } else {
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
  if (volume > 0) { cub_local_sort_inplace<VAL>(inptr, argptr, volume, sort_dim_size, stream); }
}

template <LegateTypeCode CODE, std::enable_if_t<!support_cub<CODE>::value>* = nullptr>
void local_sort_inplace(legate_type_of<CODE>* inptr,
                        int64_t* argptr,
                        const size_t volume,
                        const size_t sort_dim_size,
                        cudaStream_t stream)
{
  using VAL = legate_type_of<CODE>;
  if (volume > 0) { thrust_local_sort_inplace<VAL>(inptr, argptr, volume, sort_dim_size, stream); }
}

// auto align to multiples of 16 bytes
auto get_aligned_size = [](auto size) { return std::max<size_t>(16, (size + 15) / 16 * 16); };

template <typename VAL>
struct SortPiece {
  Buffer<VAL> values;
  Buffer<int64_t> indices;
  size_t size;
};

template <typename VAL>
struct Sample {
  VAL value;
  int32_t rank;
  size_t position;
};

template <typename VAL>
struct SampleComparator : public thrust::binary_function<Sample<VAL>, Sample<VAL>, bool> {
  __host__ __device__ bool operator()(const Sample<VAL>& lhs, const Sample<VAL>& rhs) const
  {
    // special case for unused samples
    if (lhs.rank < 0 || rhs.rank < 0) { return rhs.rank < 0 && lhs.rank >= 0; }

    if (lhs.value != rhs.value) {
      return lhs.value < rhs.value;
    } else if (lhs.rank != rhs.rank) {
      return lhs.rank < rhs.rank;
    } else {
      return lhs.position < rhs.position;
    }
  }
};

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  extract_samples(const VAL* data,
                  const size_t volume,
                  Sample<VAL>* samples,
                  const size_t num_local_samples,
                  const Sample<VAL> init_sample,
                  const size_t offset,
                  const size_t rank)
{
  const size_t sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample_idx >= num_local_samples) return;

  if (num_local_samples < volume) {
    const size_t index                    = (sample_idx + 1) * volume / num_local_samples - 1;
    samples[offset + sample_idx].value    = data[index];
    samples[offset + sample_idx].rank     = rank;
    samples[offset + sample_idx].position = index;
  } else {
    // edge case where num_local_samples > volume
    if (sample_idx < volume) {
      samples[offset + sample_idx].value    = data[sample_idx];
      samples[offset + sample_idx].rank     = rank;
      samples[offset + sample_idx].position = sample_idx;
    } else {
      samples[offset + sample_idx] = init_sample;
    }
  }
}

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  extract_split_positions(const VAL* data,
                          const size_t volume,
                          const Sample<VAL>* samples,
                          const size_t num_samples,
                          size_t* split_positions,
                          const size_t num_splitters,
                          const size_t rank)
{
  const size_t splitter_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (splitter_idx >= num_splitters) return;

  const size_t index         = (splitter_idx + 1) * num_samples / (num_splitters + 1) - 1;
  const Sample<VAL> splitter = samples[index];

  // now perform search on data to receive position *after* last element to be
  // part of the package for rank splitter_idx
  if (rank > splitter.rank) {
    // position of the last position with smaller value than splitter.value + 1
    split_positions[splitter_idx] = cub::LowerBound(data, volume, splitter.value);
  } else if (rank < splitter.rank) {
    // position of the first position with value larger than splitter.value
    split_positions[splitter_idx] = cub::UpperBound(data, volume, splitter.value);
  } else {
    split_positions[splitter_idx] = splitter.position + 1;
  }
}

template <typename VAL>
static SortPiece<VAL> sample_sort_nccl(SortPiece<VAL> local_sorted,
                                       size_t my_rank,
                                       size_t num_ranks,
                                       bool argsort,
                                       cudaStream_t stream,
                                       ncclComm_t* comm)
{
  size_t volume = local_sorted.size;

  // collect local samples
  size_t num_local_samples  = num_ranks;  // handle case numRanks > volume!!
  size_t num_global_samples = num_local_samples * num_ranks;
  auto samples              = create_buffer<Sample<VAL>>(num_global_samples, Memory::GPU_FB_MEM);

  Sample<VAL> init_sample;
  {
    const size_t num_blocks = (num_local_samples + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    init_sample.rank        = -1;  // init samples that are not populated
    size_t offset           = num_local_samples * my_rank;
    extract_samples<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(local_sorted.values.ptr(0),
                                                                  volume,
                                                                  samples.ptr(0),
                                                                  num_local_samples,
                                                                  init_sample,
                                                                  offset,
                                                                  my_rank);
  }

  // AllGather: check alignment? as we want to receive data in-place we take exact size for now
  CHECK_NCCL(ncclAllGather(samples.ptr(my_rank * num_ranks),
                           samples.ptr(0),
                           num_ranks * sizeof(Sample<VAL>),
                           ncclInt8,
                           *comm,
                           stream));

  // sort samples on device
  thrust::stable_sort(thrust::cuda::par.on(stream),
                      samples.ptr(0),
                      samples.ptr(0) + num_global_samples,
                      SampleComparator<VAL>());

  auto lower_bound          = thrust::lower_bound(thrust::cuda::par.on(stream),
                                         samples.ptr(0),
                                         samples.ptr(0) + num_global_samples,
                                         init_sample,
                                         SampleComparator<VAL>());
  size_t num_usable_samples = lower_bound - samples.ptr(0);

  // select splitters / positions based on samples (on device)
  const size_t num_splitters = num_ranks - 1;
  auto split_positions       = create_buffer<size_t>(num_splitters, Memory::Z_COPY_MEM);
  {
    const size_t num_blocks = (num_splitters + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    VAL init_value          = std::numeric_limits<VAL>::max();
    extract_split_positions<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      local_sorted.values.ptr(0),
      volume,
      samples.ptr(0),
      num_usable_samples,
      split_positions.ptr(0),
      num_splitters,
      my_rank);
  }

  // need to sync as we share values in between host/device
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // collect sizes2send, send to rank i: local_sort_data from positions  split_positions[i-1],
  // split_positions[i] - 1
  auto size_send = create_buffer<size_t>(num_ranks, Memory::Z_COPY_MEM);
  {
    size_t last_position = 0;
    for (size_t rank = 0; rank < num_ranks - 1; ++rank) {
      size_t cur_position = split_positions[rank];
      size_send[rank]     = cur_position - last_position;
      last_position       = cur_position;
    }
    size_send[num_ranks - 1] = volume - last_position;
  }

  // need to sync as we share values in between host/device
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // all2all exchange send/receive sizes
  auto size_recv = create_buffer<size_t>(num_ranks, Memory::Z_COPY_MEM);
  CHECK_NCCL(ncclGroupStart());
  for (size_t r = 0; r < num_ranks; r++) {
    CHECK_NCCL(ncclSend(size_send.ptr(r), 1, ncclUint64, r, *comm, stream));
    CHECK_NCCL(ncclRecv(size_recv.ptr(r), 1, ncclUint64, r, *comm, stream));
  }
  CHECK_NCCL(ncclGroupEnd());

  // need to sync as we share values in between host/device
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // allocate merge targets, data transfer...
  std::vector<SortPiece<VAL>> merge_buffers(num_ranks);

  for (size_t i = 0; i < num_ranks; ++i) {
    // align buffer to allow data transfer of 16byte blocks
    auto recv_size_aligned   = get_aligned_size(size_recv[i] * sizeof(VAL));
    auto buf_size            = (recv_size_aligned + sizeof(VAL) - 1) / sizeof(VAL);
    merge_buffers[i].values  = create_buffer<VAL>(buf_size, Memory::GPU_FB_MEM);
    merge_buffers[i].indices = create_buffer<int64_t>(argsort ? buf_size : 0, Memory::GPU_FB_MEM);
    merge_buffers[i].size    = size_recv[i];
  }
  size_t send_pos = 0;
  CHECK_NCCL(ncclGroupStart());
  for (size_t r = 0; r < num_ranks; r++) {
    CHECK_NCCL(ncclSend(local_sorted.values.ptr(send_pos),
                        get_aligned_size(size_send[r] * sizeof(VAL)),
                        ncclInt8,
                        r,
                        *comm,
                        stream));
    CHECK_NCCL(ncclRecv(merge_buffers[r].values.ptr(0),
                        get_aligned_size(size_recv[r] * sizeof(VAL)),
                        ncclInt8,
                        r,
                        *comm,
                        stream));
    if (argsort) {
      CHECK_NCCL(
        ncclSend(local_sorted.indices.ptr(send_pos), size_send[r], ncclInt64, r, *comm, stream));
      CHECK_NCCL(
        ncclRecv(merge_buffers[r].indices.ptr(0), size_recv[r], ncclInt64, r, *comm, stream));
    }
    send_pos += size_send[r];
  }
  CHECK_NCCL(ncclGroupEnd());

  // now merge sort all into the result buffer
  // maybe k-way merge is more efficient here...
  for (size_t stride = 1; stride < num_ranks; stride *= 2) {
    for (size_t pos = 0; pos + stride < num_ranks; pos += 2 * stride) {
      SortPiece<VAL> source1 = merge_buffers[pos];
      SortPiece<VAL> source2 = merge_buffers[pos + stride];
      auto merged_size       = source1.size + source2.size;
      auto merged_values     = create_buffer<VAL>(merged_size);
      auto merged_indices    = source1.indices;  // will be overriden for argsort
      auto p_merged_values   = merged_values.ptr(0);
      auto p_values1         = source1.values.ptr(0);
      auto p_values2         = source2.values.ptr(0);
      if (argsort) {
        merged_indices = create_buffer<int64_t>(merged_size);
        // merge with key/value
        auto p_indices1       = source1.indices.ptr(0);
        auto p_indices2       = source2.indices.ptr(0);
        auto p_merged_indices = merged_indices.ptr(0);
        thrust::merge_by_key(thrust::cuda::par.on(stream),
                             p_values1,
                             p_values1 + source1.size,
                             p_values2,
                             p_values2 + source2.size,
                             p_indices1,
                             p_indices2,
                             p_merged_values,
                             p_merged_indices);
        source1.indices.destroy();
      } else {
        thrust::merge(thrust::cuda::par.on(stream),
                      p_values1,
                      p_values1 + source1.size,
                      p_values2,
                      p_values2 + source2.size,
                      p_merged_values);
      }

      source1.values.destroy();
      source2.values.destroy();
      source2.indices.destroy();

      merge_buffers[pos].values  = merged_values;
      merge_buffers[pos].indices = merged_indices;
      merge_buffers[pos].size    = merged_size;
    }
  }
  return merge_buffers[0];
}

template <LegateTypeCode CODE, int32_t DIM>
struct SortImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(const Array& input_array,
                  Array& output_array,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t volume,
                  const size_t sort_dim_size,
                  const bool argsort,
                  const bool stable,
                  const bool is_index_space,
                  const size_t local_rank,
                  const size_t num_ranks,
                  const std::vector<comm::Communicator>& comms)
  {
    auto input = input_array.read_accessor<VAL, DIM>(rect);

    // we allow empty domains for distributed sorting
    assert(rect.empty() || input.accessor.is_dense_row_major(rect));

    auto stream = get_cached_stream();

    // make a copy of the input
    auto dense_input_copy = create_buffer<VAL>(volume, Legion::Memory::Kind::GPU_FB_MEM);
    CHECK_CUDA(cudaMemcpyAsync(dense_input_copy.ptr(0),
                               input.ptr(rect.lo),
                               sizeof(VAL) * volume,
                               cudaMemcpyDeviceToDevice,
                               stream));

    // we need a buffer for argsort
    auto indices_buffer =
      create_buffer<int64_t>(argsort ? volume : 0, Legion::Memory::Kind::GPU_FB_MEM);
    if (argsort && volume > 0) {
      // intialize
      if (DIM == 1) {
        size_t offset = DIM > 1 ? 0 : rect.lo[0];
        thrust::sequence(thrust::cuda::par.on(stream),
                         indices_buffer.ptr(0),
                         indices_buffer.ptr(0) + volume,
                         offset);
      } else {
        thrust::transform(thrust::cuda::par.on(stream),
                          thrust::make_counting_iterator<int64_t>(0),
                          thrust::make_counting_iterator<int64_t>(volume),
                          thrust::make_constant_iterator<int64_t>(sort_dim_size),
                          indices_buffer.ptr(0),
                          thrust::modulus<int64_t>());
      }
    }

    // sort data
    local_sort_inplace<CODE>(dense_input_copy.ptr(0),
                             argsort ? indices_buffer.ptr(0) : nullptr,
                             volume,
                             sort_dim_size,
                             stream);

    // this is linked to the decision in sorting.py on when to use adn 'unbounded' output array.
    if (output_array.dim() == -1) {
      SortPiece<VAL> local_sorted;
      local_sorted.values  = dense_input_copy;
      local_sorted.indices = indices_buffer;
      local_sorted.size    = volume;
      SortPiece<VAL> local_sorted_repartitioned =
        is_index_space
          ? sample_sort_nccl(
              local_sorted, local_rank, num_ranks, argsort, stream, comms[0].get<ncclComm_t*>())
          : local_sorted;
      if (argsort) {
        output_array.return_data(local_sorted_repartitioned.indices,
                                 local_sorted_repartitioned.size);
      } else {
        output_array.return_data(local_sorted_repartitioned.values,
                                 local_sorted_repartitioned.size);
      }
    } else {
      // copy back data (we assume output partition to be aliged to input!)
      if (argsort) {
        AccessorWO<int64_t, DIM> output = output_array.write_accessor<int64_t, DIM>(rect);
        assert(output.accessor.is_dense_row_major(rect));
        CHECK_CUDA(cudaMemcpyAsync(output.ptr(rect.lo),
                                   indices_buffer.ptr(0),
                                   sizeof(int64_t) * volume,
                                   cudaMemcpyDeviceToDevice,
                                   stream));
      } else {
        AccessorWO<VAL, DIM> output = output_array.write_accessor<VAL, DIM>(rect);
        assert(output.accessor.is_dense_row_major(rect));
        CHECK_CUDA(cudaMemcpyAsync(output.ptr(rect.lo),
                                   dense_input_copy.ptr(0),
                                   sizeof(VAL) * volume,
                                   cudaMemcpyDeviceToDevice,
                                   stream));
      }
    }
  }
};

/*static*/ void SortTask::gpu_variant(TaskContext& context)
{
  sort_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
