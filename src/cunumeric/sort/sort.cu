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
#include "cunumeric/utilities/thrust_allocator.h"

#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/thread/thread_search.cuh>

#include "cunumeric/cuda_help.h"

// above this threshold segment sort will be performed
// by cub::DeviceSegmentedRadixSort instead of thrust::(stable_)sort
// with tuple keys (not available for complex)
#define SEGMENT_THRESHOLD_RADIX_SORT 400

namespace cunumeric {

using namespace Legion;

struct multiply : public thrust::unary_function<size_t, size_t> {
  const size_t constant;

  multiply(size_t _constant) : constant(_constant) {}

  __host__ __device__ size_t operator()(const size_t& input) const { return input * constant; }
};

struct modulusWithOffset {
  const size_t constant;

  modulusWithOffset(size_t _constant) : constant(_constant) {}

  __host__ __device__ int64_t operator()(const int64_t& lhs, const int64_t& rhs) const
  {
    return lhs % rhs + constant;
  }
};

template <class VAL>
void cub_local_sort(const VAL* values_in,
                    VAL* values_out,
                    const int64_t* indices_in,
                    int64_t* indices_out,
                    const size_t volume,
                    const size_t sort_dim_size,
                    cudaStream_t stream)
{
  Buffer<VAL> keys_in;
  const VAL* values_in_cub = values_in;
  if (values_in == values_out) {
    keys_in       = create_buffer<VAL>(volume, Legion::Memory::Kind::GPU_FB_MEM);
    values_in_cub = keys_in.ptr(0);
    CHECK_CUDA(cudaMemcpyAsync(
      keys_in.ptr(0), values_out, sizeof(VAL) * volume, cudaMemcpyDeviceToDevice, stream));
  }

  size_t temp_storage_bytes = 0;
  if (indices_out == nullptr) {
    if (volume == sort_dim_size) {
      // sort (initial call to compute buffer size)
      cub::DeviceRadixSort::SortKeys(
        nullptr, temp_storage_bytes, values_in_cub, values_out, volume, 0, sizeof(VAL) * 8, stream);
      auto temp_storage =
        create_buffer<unsigned char>(temp_storage_bytes, Legion::Memory::Kind::GPU_FB_MEM);
      cub::DeviceRadixSort::SortKeys(temp_storage.ptr(0),
                                     temp_storage_bytes,
                                     values_in_cub,
                                     values_out,
                                     volume,
                                     0,
                                     sizeof(VAL) * 8,
                                     stream);
      temp_storage.destroy();
    } else {
      // segmented sort (initial call to compute buffer size)
      // generate start/end positions for all segments via iterators to avoid allocating buffers
      auto off_start_pos_it = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_t>(0), multiply(sort_dim_size));
      auto off_end_pos_it = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_t>(1), multiply(sort_dim_size));

      cub::DeviceSegmentedRadixSort::SortKeys(nullptr,
                                              temp_storage_bytes,
                                              values_in_cub,
                                              values_out,
                                              volume,
                                              volume / sort_dim_size,
                                              off_start_pos_it,
                                              off_end_pos_it,
                                              0,
                                              sizeof(VAL) * 8,
                                              stream);
      auto temp_storage =
        create_buffer<unsigned char>(temp_storage_bytes, Legion::Memory::Kind::GPU_FB_MEM);

      cub::DeviceSegmentedRadixSort::SortKeys(temp_storage.ptr(0),
                                              temp_storage_bytes,
                                              values_in_cub,
                                              values_out,
                                              volume,
                                              volume / sort_dim_size,
                                              off_start_pos_it,
                                              off_end_pos_it,
                                              0,
                                              sizeof(VAL) * 8,
                                              stream);
      temp_storage.destroy();
    }
  } else {
    Buffer<int64_t> idx_in;
    const int64_t* indices_in_cub = indices_in;
    if (indices_in == indices_out) {
      idx_in         = create_buffer<int64_t>(volume, Legion::Memory::Kind::GPU_FB_MEM);
      indices_in_cub = idx_in.ptr(0);
      CHECK_CUDA(cudaMemcpyAsync(
        idx_in.ptr(0), indices_out, sizeof(int64_t) * volume, cudaMemcpyDeviceToDevice, stream));
    }

    if (volume == sort_dim_size) {
      // argsort (initial call to compute buffer size)
      cub::DeviceRadixSort::SortPairs(nullptr,
                                      temp_storage_bytes,
                                      values_in_cub,
                                      values_out,
                                      indices_in_cub,
                                      indices_out,
                                      volume,
                                      0,
                                      sizeof(VAL) * 8,
                                      stream);

      auto temp_storage =
        create_buffer<unsigned char>(temp_storage_bytes, Legion::Memory::Kind::GPU_FB_MEM);

      cub::DeviceRadixSort::SortPairs(temp_storage.ptr(0),
                                      temp_storage_bytes,
                                      values_in_cub,
                                      values_out,
                                      indices_in_cub,
                                      indices_out,
                                      volume,
                                      0,
                                      sizeof(VAL) * 8,
                                      stream);
      temp_storage.destroy();
    } else {
      // segmented argsort (initial call to compute buffer size)
      // generate start/end positions for all segments via iterators to avoid allocating buffers
      auto off_start_pos_it = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_t>(0), multiply(sort_dim_size));
      auto off_end_pos_it = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_t>(1), multiply(sort_dim_size));

      cub::DeviceSegmentedRadixSort::SortPairs(nullptr,
                                               temp_storage_bytes,
                                               values_in_cub,
                                               values_out,
                                               indices_in_cub,
                                               indices_out,
                                               volume,
                                               volume / sort_dim_size,
                                               off_start_pos_it,
                                               off_end_pos_it,
                                               0,
                                               sizeof(VAL) * 8,
                                               stream);

      auto temp_storage =
        create_buffer<unsigned char>(temp_storage_bytes, Legion::Memory::Kind::GPU_FB_MEM);

      cub::DeviceSegmentedRadixSort::SortPairs(temp_storage.ptr(0),
                                               temp_storage_bytes,
                                               values_in_cub,
                                               values_out,
                                               indices_in_cub,
                                               indices_out,
                                               volume,
                                               volume / sort_dim_size,
                                               off_start_pos_it,
                                               off_end_pos_it,
                                               0,
                                               sizeof(VAL) * 8,
                                               stream);
      temp_storage.destroy();
    }
    if (indices_in == indices_out) idx_in.destroy();
  }

  if (values_in == values_out) keys_in.destroy();
}

template <class VAL>
void thrust_local_sort(const VAL* values_in,
                       VAL* values_out,
                       const int64_t* indices_in,
                       int64_t* indices_out,
                       const size_t volume,
                       const size_t sort_dim_size,
                       const bool stable,
                       cudaStream_t stream)
{
  auto alloc       = ThrustAllocator(Memory::GPU_FB_MEM);
  auto exec_policy = thrust::cuda::par(alloc).on(stream);

  if (values_in != values_out) {
    // not in-place --> need a copy
    CHECK_CUDA(cudaMemcpyAsync(
      values_out, values_in, sizeof(VAL) * volume, cudaMemcpyDeviceToDevice, stream));
  }
  if (indices_in != indices_out) {
    // not in-place --> need a copy
    CHECK_CUDA(cudaMemcpyAsync(
      indices_out, values_in, sizeof(int64_t) * volume, cudaMemcpyDeviceToDevice, stream));
  }

  if (indices_out == nullptr) {
    if (volume == sort_dim_size) {
      if (stable) {
        thrust::stable_sort(exec_policy, values_out, values_out + volume);
      } else {
        thrust::sort(exec_policy, values_out, values_out + volume);
      }
    } else {
      auto sort_id = create_buffer<uint64_t>(volume, Legion::Memory::Kind::GPU_FB_MEM);
      // init combined keys
      thrust::transform(exec_policy,
                        thrust::make_counting_iterator<uint64_t>(0),
                        thrust::make_counting_iterator<uint64_t>(volume),
                        thrust::make_constant_iterator<uint64_t>(sort_dim_size),
                        sort_id.ptr(0),
                        thrust::divides<uint64_t>());
      auto combined = thrust::make_zip_iterator(thrust::make_tuple(sort_id.ptr(0), values_out));

      if (stable) {
        thrust::stable_sort(
          exec_policy, combined, combined + volume, thrust::less<thrust::tuple<size_t, VAL>>());
      } else {
        thrust::sort(
          exec_policy, combined, combined + volume, thrust::less<thrust::tuple<size_t, VAL>>());
      }

      sort_id.destroy();
    }
  } else {
    if (volume == sort_dim_size) {
      if (stable) {
        thrust::stable_sort_by_key(exec_policy, values_out, values_out + volume, indices_out);
      } else {
        thrust::sort_by_key(exec_policy, values_out, values_out + volume, indices_out);
      }
    } else {
      auto sort_id = create_buffer<uint64_t>(volume, Legion::Memory::Kind::GPU_FB_MEM);
      // init combined keys
      thrust::transform(exec_policy,
                        thrust::make_counting_iterator<uint64_t>(0),
                        thrust::make_counting_iterator<uint64_t>(volume),
                        thrust::make_constant_iterator<uint64_t>(sort_dim_size),
                        sort_id.ptr(0),
                        thrust::divides<uint64_t>());
      auto combined = thrust::make_zip_iterator(thrust::make_tuple(sort_id.ptr(0), values_out));

      if (stable) {
        thrust::stable_sort_by_key(exec_policy,
                                   combined,
                                   combined + volume,
                                   indices_out,
                                   thrust::less<thrust::tuple<size_t, VAL>>());
      } else {
        thrust::sort_by_key(exec_policy,
                            combined,
                            combined + volume,
                            indices_out,
                            thrust::less<thrust::tuple<size_t, VAL>>());
      }

      sort_id.destroy();
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
void local_sort(const legate_type_of<CODE>* values_in,
                legate_type_of<CODE>* values_out,
                const int64_t* indices_in,
                int64_t* indices_out,
                const size_t volume,
                const size_t sort_dim_size,
                const bool stable,  // cub sort is always stable
                cudaStream_t stream)
{
  using VAL = legate_type_of<CODE>;
  // fallback to thrust approach as segmented radix sort is not suited for small segments
  if (volume == sort_dim_size || sort_dim_size > SEGMENT_THRESHOLD_RADIX_SORT) {
    cub_local_sort<VAL>(
      values_in, values_out, indices_in, indices_out, volume, sort_dim_size, stream);
  } else {
    thrust_local_sort<VAL>(
      values_in, values_out, indices_in, indices_out, volume, sort_dim_size, stable, stream);
  }
}

template <LegateTypeCode CODE, std::enable_if_t<!support_cub<CODE>::value>* = nullptr>
void local_sort(const legate_type_of<CODE>* values_in,
                legate_type_of<CODE>* values_out,
                const int64_t* indices_in,
                int64_t* indices_out,
                const size_t volume,
                const size_t sort_dim_size,
                const bool stable,
                cudaStream_t stream)
{
  using VAL = legate_type_of<CODE>;
  thrust_local_sort<VAL>(
    values_in, values_out, indices_in, indices_out, volume, sort_dim_size, stable, stream);
}

// auto align to multiples of 16 bytes
auto get_16b_aligned = [](auto bytes) { return std::max<size_t>(16, (bytes + 15) / 16 * 16); };
auto get_16b_aligned_count = [](auto count, auto element_bytes) {
  return (get_16b_aligned(count * element_bytes) + element_bytes - 1) / element_bytes;
};

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

  // collect local samples - for now we take num_ranks samples for every node
  // worst case this leads to 2*N/ranks elements on a single node
  size_t num_local_samples = num_ranks;

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
  auto alloc       = ThrustAllocator(Memory::GPU_FB_MEM);
  auto exec_policy = thrust::cuda::par(alloc).on(stream);
  thrust::stable_sort(
    exec_policy, samples.ptr(0), samples.ptr(0) + num_global_samples, SampleComparator<VAL>());

  auto lower_bound          = thrust::lower_bound(exec_policy,
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

  // cleanup intermediate data structures
  samples.destroy();
  split_positions.destroy();

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

  // handle alignment
  std::vector<size_t> aligned_pos_vals_send(num_ranks);
  std::vector<size_t> aligned_pos_idcs_send(num_ranks);
  size_t buf_size_send_vals_total = 0;
  size_t buf_size_send_idcs_total = 0;
  for (size_t i = 0; i < num_ranks; ++i) {
    // align buffer to allow data transfer of 16byte blocks
    aligned_pos_vals_send[i] = buf_size_send_vals_total;
    buf_size_send_vals_total += get_16b_aligned_count(size_send[i], sizeof(VAL));
    if (argsort) {
      aligned_pos_idcs_send[i] = buf_size_send_idcs_total;
      buf_size_send_idcs_total += get_16b_aligned_count(size_send[i], sizeof(int64_t));
    }
  }

  // copy values into aligned send buffer
  auto val_send_buf = local_sorted.values;
  if (buf_size_send_vals_total > volume) {
    val_send_buf = create_buffer<VAL>(buf_size_send_vals_total, Memory::GPU_FB_MEM);
    size_t pos   = 0;
    for (size_t r = 0; r < num_ranks; ++r) {
      CHECK_CUDA(cudaMemcpyAsync(val_send_buf.ptr(aligned_pos_vals_send[r]),
                                 local_sorted.values.ptr(pos),
                                 sizeof(VAL) * size_send[r],
                                 cudaMemcpyDeviceToDevice,
                                 stream));
      pos += size_send[r];
    }
    local_sorted.values.destroy();
  }

  // copy indices into aligned send buffer
  auto idc_send_buf = local_sorted.indices;
  if (argsort && buf_size_send_idcs_total > volume) {
    idc_send_buf = create_buffer<int64_t>(buf_size_send_idcs_total, Memory::GPU_FB_MEM);
    size_t pos   = 0;
    for (size_t r = 0; r < num_ranks; ++r) {
      CHECK_CUDA(cudaMemcpyAsync(idc_send_buf.ptr(aligned_pos_idcs_send[r]),
                                 local_sorted.indices.ptr(pos),
                                 sizeof(int64_t) * size_send[r],
                                 cudaMemcpyDeviceToDevice,
                                 stream));
      pos += size_send[r];
    }
    local_sorted.indices.destroy();
  }

  // allocate target buffers
  std::vector<SortPiece<VAL>> merge_buffers(num_ranks);
  for (size_t i = 0; i < num_ranks; ++i) {
    auto buf_size_vals_recv = get_16b_aligned_count(size_recv[i], sizeof(VAL));
    merge_buffers[i].values = create_buffer<VAL>(buf_size_vals_recv, Memory::GPU_FB_MEM);
    merge_buffers[i].size   = size_recv[i];
    if (argsort) {
      auto buf_size_idcs_recv  = get_16b_aligned_count(size_recv[i], sizeof(int64_t));
      merge_buffers[i].indices = create_buffer<int64_t>(buf_size_idcs_recv, Memory::GPU_FB_MEM);
    } else {
      merge_buffers[i].indices = create_buffer<int64_t>(0, Memory::GPU_FB_MEM);
    }
  }
  CHECK_NCCL(ncclGroupStart());
  for (size_t r = 0; r < num_ranks; r++) {
    CHECK_NCCL(ncclSend(val_send_buf.ptr(aligned_pos_vals_send[r]),
                        get_16b_aligned(size_send[r] * sizeof(VAL)),
                        ncclInt8,
                        r,
                        *comm,
                        stream));
    CHECK_NCCL(ncclRecv(merge_buffers[r].values.ptr(0),
                        get_16b_aligned(size_recv[r] * sizeof(VAL)),
                        ncclInt8,
                        r,
                        *comm,
                        stream));
  }
  CHECK_NCCL(ncclGroupEnd());

  if (argsort) {
    CHECK_NCCL(ncclGroupStart());
    for (size_t r = 0; r < num_ranks; r++) {
      CHECK_NCCL(ncclSend(idc_send_buf.ptr(aligned_pos_idcs_send[r]),
                          get_16b_aligned_count(size_send[r], sizeof(int64_t)),
                          ncclInt64,
                          r,
                          *comm,
                          stream));
      CHECK_NCCL(ncclRecv(merge_buffers[r].indices.ptr(0),
                          get_16b_aligned_count(size_recv[r], sizeof(int64_t)),
                          ncclInt64,
                          r,
                          *comm,
                          stream));
    }
    CHECK_NCCL(ncclGroupEnd());
  }

  // cleanup remaining buffers
  size_send.destroy();
  size_recv.destroy();
  val_send_buf.destroy();
  idc_send_buf.destroy();

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
        thrust::merge_by_key(exec_policy,
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
        thrust::merge(exec_policy,
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

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename VAL>
struct SegmentSample {
  VAL value;
  size_t segment;
  int32_t rank;
  size_t position;
};

template <typename VAL>
struct SegmentMergePiece {
  Buffer<size_t> segments;
  Buffer<VAL> values;
  Buffer<int64_t> indices;
  size_t size;
};

template <typename VAL>
struct SegmentSampleComparator
  : public thrust::binary_function<SegmentSample<VAL>, SegmentSample<VAL>, bool> {
  __host__ __device__ bool operator()(const SegmentSample<VAL>& lhs,
                                      const SegmentSample<VAL>& rhs) const
  {
    if (lhs.segment != rhs.segment) {
      return lhs.segment < rhs.segment;
    } else {
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
  }
};

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  extract_split_positions_segments(const VAL* data,
                                   const size_t segment_size_l,
                                   const SegmentSample<VAL>* samples,
                                   const size_t num_segments_l,
                                   const size_t num_samples_per_segment,
                                   const size_t num_usable_samples_per_segment,
                                   size_t* split_positions,
                                   const size_t num_splitters,
                                   const size_t my_sort_rank)
{
  const size_t splitter_idx_g = blockIdx.x * blockDim.x + threadIdx.x;
  if (splitter_idx_g >= num_splitters) return;

  const size_t num_splitters_per_segment = num_splitters / num_segments_l;
  const size_t splitter_pos              = splitter_idx_g % num_splitters_per_segment;
  const size_t splitter_segment          = splitter_idx_g / num_splitters_per_segment;

  const size_t index =
    (splitter_pos + 1) * num_usable_samples_per_segment / (num_splitters_per_segment + 1) - 1;
  const SegmentSample<VAL> splitter = samples[splitter_segment * num_samples_per_segment + index];

  // now perform search on data to receive position *after* last element to be
  // part of the package for my_sort_rank splitter_idx_g
  const size_t offset = splitter_segment * segment_size_l;
  if (my_sort_rank > splitter.rank) {
    // position of the last position with smaller value than splitter.value + 1
    split_positions[splitter_idx_g] =
      cub::LowerBound(data + offset, segment_size_l, splitter.value) + offset;
  } else if (my_sort_rank < splitter.rank) {
    // position of the first position with value larger than splitter.value
    split_positions[splitter_idx_g] =
      cub::UpperBound(data + offset, segment_size_l, splitter.value) + offset;
  } else {
    split_positions[splitter_idx_g] = splitter.position + 1;
  }
}

__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  initialize_segment_start_positions(const size_t* start_positions,
                                     const size_t num_segments_l,
                                     size_t* segment_ids,
                                     const size_t num_segment_ids)
{
  const size_t segment_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (segment_idx >= num_segments_l) return;

  unsigned long long int* ptr = (unsigned long long int*)segment_ids;

  const size_t position = start_positions[segment_idx];
  if (position < num_segment_ids) atomicAdd(&(ptr[position]), (unsigned long long int)1l);
}

__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  extract_segment_sizes(const size_t* segments,
                        const size_t size,
                        int64_t* segments_diff,
                        const size_t num_segments_l,
                        const size_t segments_size_l)
{
  const size_t segment_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (segment_idx >= num_segments_l) return;

  const size_t position = cub::LowerBound(segments, size, segment_idx);
  const size_t next_position =
    cub::LowerBound(segments + position, size - position, segment_idx + 1) + position;

  const size_t segment_size  = next_position - position;
  segments_diff[segment_idx] = segment_size - segments_size_l;
}

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  extract_samples_segment(const VAL* data,
                          const size_t volume,
                          SegmentSample<VAL>* samples,
                          const size_t num_samples_per_segment_l,
                          const size_t segment_size_l,
                          const size_t offset,
                          const size_t num_sort_ranks,
                          const size_t sort_rank)
{
  const size_t sample_idx     = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t num_segments_l = volume / segment_size_l;
  const size_t num_samples_l  = num_samples_per_segment_l * num_segments_l;
  if (sample_idx >= num_samples_l) return;

  const size_t segment_id_l       = sample_idx / num_samples_per_segment_l;
  const size_t segment_sample_idx = sample_idx % num_samples_per_segment_l;
  const size_t sample_index       = offset + sample_idx;

  if (num_samples_per_segment_l < segment_size_l) {
    const size_t index = segment_id_l * segment_size_l +
                         (segment_sample_idx + 1) * segment_size_l / num_samples_per_segment_l - 1;
    samples[sample_index].value    = data[index];
    samples[sample_index].rank     = sort_rank;
    samples[sample_index].segment  = segment_id_l;
    samples[sample_index].position = index;
  } else {
    // edge case where num_samples_l > volume
    if (segment_sample_idx < segment_size_l) {
      const size_t index             = segment_id_l * segment_size_l + segment_sample_idx;
      samples[sample_index].value    = data[index];
      samples[sample_index].rank     = sort_rank;
      samples[sample_index].segment  = segment_id_l;
      samples[sample_index].position = index;
    } else {
      samples[sample_index].rank    = -1;  // not populated
      samples[sample_index].segment = segment_id_l;
    }
  }
}

struct subtract : public thrust::unary_function<int64_t, int64_t> {
  const int64_t constant_;

  subtract(int64_t constant) : constant_(constant) {}

  __host__ __device__ int64_t operator()(const int64_t& input) const { return input - constant_; }
};

struct positive_value : public thrust::unary_function<int64_t, int64_t> {
  __host__ __device__ int64_t operator()(const int64_t& x) const { return x > 0 ? x : 0; }
};

struct negative_value : public thrust::unary_function<int64_t, int64_t> {
  __host__ __device__ int64_t operator()(const int64_t& x) const { return x < 0 ? -x : 0; }
};

template <typename VAL>
SortPiece<VAL> sample_sort_nccl_nd(
  SortPiece<VAL> local_sorted,
  /* global domain information */
  size_t my_rank,  // global NCCL rank
  size_t num_ranks,
  size_t segment_size_g,
  /* domain information in sort dimension */
  size_t my_sort_rank,    // local rank id in sort dimension
  size_t num_sort_ranks,  // #ranks that share a sort dimension
  size_t* sort_ranks,     // rank ids that share a sort dimension with us
  size_t segment_size_l,  // (local) segment size
  /* other */
  bool argsort,
  cudaStream_t stream,
  ncclComm_t* comm)
{
  size_t volume = local_sorted.size;

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////// Part 0: detection of empty nodes
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // first of all we need to check for processes that don't want
  // to take part in the computation. This might lead to a reduction of
  // sort ranks. Note that if segment_size_l>0 && volume==0 means that we have
  // a full sort group being empty, this should not affect local sort rank size.
  {
    auto worker_count      = create_buffer<int32_t>(num_ranks, Memory::Z_COPY_MEM);
    auto* worker_count_ptr = worker_count.ptr(0);
    worker_count_ptr[0]    = segment_size_l > 0 ? 1 : 0;
    CHECK_NCCL(
      ncclAllReduce(worker_count_ptr, worker_count_ptr, 1, ncclInt32, ncclSum, *comm, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    if (worker_count_ptr[0] < num_ranks) {
      const size_t number_sort_groups = num_ranks / num_sort_ranks;
      num_sort_ranks                  = worker_count_ptr[0] / number_sort_groups;
    }
    worker_count.destroy();

    // early out
    if (volume == 0) return local_sorted;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////// Part 1: select and share samples accross sort domain
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // collect local samples - for now we take num_sort_ranks samples for every node/line
  // worst case this leads to imbalance of x2
  size_t num_segments_l            = volume / segment_size_l;
  size_t num_samples_per_segment_l = num_sort_ranks;
  size_t num_samples_l             = num_samples_per_segment_l * num_segments_l;
  size_t num_samples_per_segment_g = num_samples_per_segment_l * num_sort_ranks;
  size_t num_samples_g             = num_samples_per_segment_g * num_segments_l;
  auto samples = create_buffer<SegmentSample<VAL>>(num_samples_g, Memory::GPU_FB_MEM);

  size_t offset = num_samples_l * my_sort_rank;
  {
    const size_t num_blocks = (num_samples_l + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    extract_samples_segment<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      local_sorted.values.ptr(0),
      volume,
      samples.ptr(0),
      num_samples_per_segment_l,
      segment_size_l,
      offset,
      num_sort_ranks,
      my_sort_rank);
  }

  // AllGather does not work here as not all have the same amount!
  // This is all2all restricted to one sort row
  {
    // allocate receive buffer
    const size_t aligned_count = get_16b_aligned_count(num_samples_l, sizeof(SegmentSample<VAL>));
    auto send_buffer = create_buffer<SegmentSample<VAL>>(aligned_count, Memory::GPU_FB_MEM);
    CHECK_CUDA(cudaMemcpyAsync(send_buffer.ptr(0),
                               samples.ptr(offset),
                               sizeof(SegmentSample<VAL>) * num_samples_l,
                               cudaMemcpyDeviceToDevice,
                               stream));

    auto recv_buffer =
      create_buffer<SegmentSample<VAL>>(aligned_count * num_sort_ranks, Memory::GPU_FB_MEM);

    CHECK_NCCL(ncclGroupStart());
    for (size_t r = 0; r < num_sort_ranks; r++) {
      if (r != my_sort_rank) {
        CHECK_NCCL(ncclSend(send_buffer.ptr(0),
                            num_samples_l * sizeof(SegmentSample<VAL>),
                            ncclInt8,
                            sort_ranks[r],
                            *comm,
                            stream));
        CHECK_NCCL(ncclRecv(recv_buffer.ptr(aligned_count * r),
                            num_samples_l * sizeof(SegmentSample<VAL>),
                            ncclInt8,
                            sort_ranks[r],
                            *comm,
                            stream));
      }
    }
    CHECK_NCCL(ncclGroupEnd());

    // copy back
    for (size_t r = 0; r < num_sort_ranks; r++) {
      if (r != my_sort_rank) {
        CHECK_CUDA(cudaMemcpyAsync(samples.ptr(num_samples_l * r),
                                   recv_buffer.ptr(aligned_count * r),
                                   sizeof(SegmentSample<VAL>) * num_samples_l,
                                   cudaMemcpyDeviceToDevice,
                                   stream));
      }
    }

    // destroy
    send_buffer.destroy();
    recv_buffer.destroy();
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////// Part 2: select splitters from samples and collect positions in local data
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // sort samples on device
  auto alloc       = ThrustAllocator(Memory::GPU_FB_MEM);
  auto exec_policy = thrust::cuda::par(alloc).on(stream);
  thrust::stable_sort(
    exec_policy, samples.ptr(0), samples.ptr(0) + num_samples_g, SegmentSampleComparator<VAL>());

  // check whether we have invalid samples (in case one participant did not have enough)
  SegmentSample<VAL> invalid_sample;
  invalid_sample.segment                = 0;
  invalid_sample.rank                   = -1;
  auto lower_bound                      = thrust::lower_bound(exec_policy,
                                         samples.ptr(0),
                                         samples.ptr(0) + num_samples_per_segment_g,
                                         invalid_sample,
                                         SegmentSampleComparator<VAL>());
  size_t num_usable_samples_per_segment = lower_bound - samples.ptr(0);

  // select splitters / positions based on samples (on device)
  // the indexing is split_positions[segments][positions]
  const size_t num_splitters = (num_sort_ranks - 1) * num_segments_l;
  auto split_positions       = create_buffer<size_t>(num_splitters, Memory::Z_COPY_MEM);
  {
    const size_t num_blocks = (num_splitters + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    extract_split_positions_segments<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      local_sorted.values.ptr(0),
      segment_size_l,
      samples.ptr(0),
      num_segments_l,
      num_samples_per_segment_g,
      num_usable_samples_per_segment,
      split_positions.ptr(0),
      num_splitters,
      my_sort_rank);
  }

  // need to sync as we share values in between host/device
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // initialize sizes to send
  std::vector<Buffer<size_t>> size_send(num_sort_ranks);
  for (size_t r = 0; r < num_sort_ranks; r++) {
    size_send[r] =
      create_buffer<size_t>(num_segments_l + 1, Memory::Z_COPY_MEM);  // last element stores sum
    size_send[r][num_segments_l] = 0;
  }
  for (size_t segment = 0; segment < num_segments_l; ++segment) {
    size_t last_position = segment_size_l * segment;  // positions global
    for (size_t r = 0; r < num_sort_ranks - 1; r++) {
      size_t cur_position   = split_positions[segment * (num_sort_ranks - 1) + r];
      size_send[r][segment] = cur_position - last_position;
      size_send[r][num_segments_l] += size_send[r][segment];
      last_position = cur_position;
    }
    {
      // last rank separately
      const size_t size                      = (segment + 1) * segment_size_l - last_position;
      size_send[num_sort_ranks - 1][segment] = size;
      size_send[num_sort_ranks - 1][num_segments_l] += size;
    }
  }

  // cleanup intermediate data structures
  samples.destroy();
  split_positions.destroy();

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////// Part 3: communicate data in sort domain
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // all2all exchange send/receive sizes
  std::vector<Buffer<size_t>> size_recv(num_sort_ranks);
  for (size_t r = 0; r < num_sort_ranks; r++) {
    size_recv[r] =
      create_buffer<size_t>(num_segments_l + 1, Memory::Z_COPY_MEM);  // last element stores sum
  }
  CHECK_NCCL(ncclGroupStart());
  for (size_t r = 0; r < num_sort_ranks; r++) {
    CHECK_NCCL(
      ncclSend(size_send[r].ptr(0), num_segments_l + 1, ncclUint64, sort_ranks[r], *comm, stream));
    CHECK_NCCL(
      ncclRecv(size_recv[r].ptr(0), num_segments_l + 1, ncclUint64, sort_ranks[r], *comm, stream));
  }
  CHECK_NCCL(ncclGroupEnd());

  // need to sync as we share values in between host/device
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // copy values into aligned send buffer
  std::vector<Buffer<VAL>> val_send_buffers(num_sort_ranks);
  std::vector<Buffer<size_t>> idc_send_buffers(num_sort_ranks);
  {
    std::vector<size_t> positions(num_sort_ranks);
    for (size_t r = 0; r < num_sort_ranks; r++) {
      val_send_buffers[r] = create_buffer<VAL>(size_send[r][num_segments_l], Memory::GPU_FB_MEM);
      if (argsort) {
        idc_send_buffers[r] =
          create_buffer<size_t>(size_send[r][num_segments_l], Memory::GPU_FB_MEM);
      }
      positions[r] = 0;
    }
    for (size_t segment = 0; segment < num_segments_l; ++segment) {
      size_t start_position = segment * segment_size_l;
      for (size_t r = 0; r < num_sort_ranks; r++) {
        size_t size = size_send[r][segment];
        CHECK_CUDA(cudaMemcpyAsync(val_send_buffers[r].ptr(positions[r]),
                                   local_sorted.values.ptr(start_position),
                                   sizeof(VAL) * size,
                                   cudaMemcpyDeviceToDevice,
                                   stream));
        if (argsort) {
          CHECK_CUDA(cudaMemcpyAsync(idc_send_buffers[r].ptr(positions[r]),
                                     local_sorted.indices.ptr(start_position),
                                     sizeof(size_t) * size,
                                     cudaMemcpyDeviceToDevice,
                                     stream));
        }
        start_position += size;
        positions[r] += size;

        assert(segment < num_segments_l - 1 || positions[r] == size_send[r][num_segments_l]);
      }
      assert(start_position == (segment + 1) * segment_size_l);
    }
    local_sorted.values.destroy();
    if (argsort) local_sorted.indices.destroy();
  }

  // allocate target buffers
  std::vector<SegmentMergePiece<VAL>> merge_buffers(num_sort_ranks);
  {
    for (size_t r = 0; r < num_sort_ranks; ++r) {
      auto size = size_recv[r][num_segments_l];

      merge_buffers[r].size     = size;
      merge_buffers[r].segments = create_buffer<size_t>(size, Memory::GPU_FB_MEM);

      // initialize segment information
      // TODO: maybe there is a better way?
      {
        // 0  1  2  1  3      // counts per segment to receive
        // 0  1  3  4  7
        // 0 1 2 3 4 5 6
        // 1 1 0 1 1 0 0
        // 1 2 2 3 4 4 4      // segment id for all received elements
        thrust::inclusive_scan(exec_policy,
                               size_recv[r].ptr(0),
                               size_recv[r].ptr(0) + num_segments_l + 1,
                               size_recv[r].ptr(0));
        CHECK_CUDA(cudaMemsetAsync(merge_buffers[r].segments.ptr(0), 0, size * sizeof(size_t)));
        const size_t num_blocks = (num_segments_l + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        assert(sizeof(unsigned long long int) ==
               sizeof(size_t));  // kernel needs to cast for atomicAdd...
        initialize_segment_start_positions<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
          size_recv[r].ptr(0),
          num_segments_l - 1,
          merge_buffers[r].segments.ptr(0),
          merge_buffers[r].size);
        thrust::inclusive_scan(exec_policy,
                               merge_buffers[r].segments.ptr(0),
                               merge_buffers[r].segments.ptr(0) + size,
                               merge_buffers[r].segments.ptr(0));
        size_recv[r].destroy();
      }

      merge_buffers[r].values = create_buffer<VAL>(size, Memory::GPU_FB_MEM);
      if (argsort) {
        merge_buffers[r].indices = create_buffer<int64_t>(size, Memory::GPU_FB_MEM);
      } else {
        merge_buffers[r].indices = create_buffer<int64_t>(0, Memory::GPU_FB_MEM);
      }
    }
  }

  // communicate all2all (in sort dimension)
  CHECK_NCCL(ncclGroupStart());
  for (size_t r = 0; r < num_sort_ranks; r++) {
    CHECK_NCCL(ncclSend(val_send_buffers[r].ptr(0),
                        size_send[r][num_segments_l] * sizeof(VAL),
                        ncclInt8,
                        sort_ranks[r],
                        *comm,
                        stream));
    CHECK_NCCL(ncclRecv(merge_buffers[r].values.ptr(0),
                        merge_buffers[r].size * sizeof(VAL),
                        ncclInt8,
                        sort_ranks[r],
                        *comm,
                        stream));
  }
  CHECK_NCCL(ncclGroupEnd());

  if (argsort) {
    CHECK_NCCL(ncclGroupStart());
    for (size_t r = 0; r < num_sort_ranks; r++) {
      CHECK_NCCL(ncclSend(idc_send_buffers[r].ptr(0),
                          size_send[r][num_segments_l],
                          ncclInt64,
                          sort_ranks[r],
                          *comm,
                          stream));
      CHECK_NCCL(ncclRecv(merge_buffers[r].indices.ptr(0),
                          merge_buffers[r].size,
                          ncclInt64,
                          sort_ranks[r],
                          *comm,
                          stream));
    }
    CHECK_NCCL(ncclGroupEnd());
  }

  // cleanup remaining buffers
  for (size_t r = 0; r < num_sort_ranks; r++) {
    size_send[r].destroy();
    // size_recv[r].destroy(); has been released before
    val_send_buffers[r].destroy();
    if (argsort) idc_send_buffers[r].destroy();
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////// Part 4: merge data
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // now merge sort all into the result buffer
  // maybe k-way merge is more efficient here...
  for (size_t stride = 1; stride < num_sort_ranks; stride *= 2) {
    for (size_t pos = 0; pos + stride < num_sort_ranks; pos += 2 * stride) {
      SegmentMergePiece<VAL> source1 = merge_buffers[pos];
      SegmentMergePiece<VAL> source2 = merge_buffers[pos + stride];
      auto merged_size               = source1.size + source2.size;
      auto merged_values             = create_buffer<VAL>(merged_size);
      auto merged_segments           = create_buffer<size_t>(merged_size);
      auto merged_indices            = source1.indices;  // will be overriden for argsort
      auto p_merged_values           = merged_values.ptr(0);
      auto p_merged_segments         = merged_segments.ptr(0);
      auto p_values1                 = source1.values.ptr(0);
      auto p_values2                 = source2.values.ptr(0);
      auto p_segments1               = source1.segments.ptr(0);
      auto p_segments2               = source2.segments.ptr(0);

      auto comb_keys_1 = thrust::make_zip_iterator(thrust::make_tuple(p_segments1, p_values1));
      auto comb_keys_2 = thrust::make_zip_iterator(thrust::make_tuple(p_segments2, p_values2));
      auto comb_keys_merged =
        thrust::make_zip_iterator(thrust::make_tuple(p_merged_segments, p_merged_values));

      if (argsort) {
        merged_indices = create_buffer<int64_t>(merged_size);
        // merge with key/value
        auto p_indices1       = source1.indices.ptr(0);
        auto p_indices2       = source2.indices.ptr(0);
        auto p_merged_indices = merged_indices.ptr(0);
        thrust::merge_by_key(exec_policy,
                             comb_keys_1,
                             comb_keys_1 + source1.size,
                             comb_keys_2,
                             comb_keys_2 + source2.size,
                             p_indices1,
                             p_indices2,
                             comb_keys_merged,
                             p_merged_indices,
                             thrust::less<thrust::tuple<size_t, VAL>>());
        source1.indices.destroy();
      } else {
        thrust::merge(exec_policy,
                      comb_keys_1,
                      comb_keys_1 + source1.size,
                      comb_keys_2,
                      comb_keys_2 + source2.size,
                      comb_keys_merged,
                      thrust::less<thrust::tuple<size_t, VAL>>());
      }

      source1.values.destroy();
      source2.values.destroy();
      source1.segments.destroy();
      source2.segments.destroy();
      source2.indices.destroy();

      merge_buffers[pos].values   = merged_values;
      merge_buffers[pos].indices  = merged_indices;
      merge_buffers[pos].segments = merged_segments;
      merge_buffers[pos].size     = merged_size;
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////// Part 5: re-balance data to match input/output dimensions
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // rebalance step
  SortPiece<VAL> result;
  {
    // compute diff for each segment
    auto segment_diff = create_buffer<int64_t>(num_segments_l, Memory::Z_COPY_MEM);
    {
      // start kernel to search from merge_buffers[0].segments
      const size_t num_blocks = (num_segments_l + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      extract_segment_sizes<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        merge_buffers[0].segments.ptr(0),
        merge_buffers[0].size,
        segment_diff.ptr(0),
        num_segments_l,
        segment_size_l);
    }

    // not needed anymore
    merge_buffers[0].segments.destroy();

    // allocate target
    std::vector<Buffer<int64_t>> segment_diff_buffers(num_sort_ranks);
    for (size_t r = 0; r < num_sort_ranks; r++) {
      segment_diff_buffers[r] = create_buffer<int64_t>(num_segments_l, Memory::GPU_FB_MEM);
    }

    // communicate segment diffs
    CHECK_NCCL(ncclGroupStart());
    for (size_t r = 0; r < num_sort_ranks; r++) {
      CHECK_NCCL(
        ncclSend(segment_diff.ptr(0), num_segments_l, ncclInt64, sort_ranks[r], *comm, stream));
      CHECK_NCCL(ncclRecv(
        segment_diff_buffers[r].ptr(0), num_segments_l, ncclInt64, sort_ranks[r], *comm, stream));
    }
    CHECK_NCCL(ncclGroupEnd());

    // copy to transpose structure [segments][ranks]
    auto segment_diff_2d =
      create_buffer<int64_t>(num_segments_l * num_sort_ranks, Memory::GPU_FB_MEM);
    for (size_t r = 0; r < num_sort_ranks; r++) {
      CHECK_CUDA(cudaMemcpy2DAsync(segment_diff_2d.ptr(r),
                                   num_sort_ranks * sizeof(int64_t),
                                   segment_diff_buffers[r].ptr(0),
                                   1 * sizeof(int64_t),
                                   sizeof(int64_t),
                                   num_segments_l,
                                   cudaMemcpyDeviceToDevice,
                                   stream));
    }
    for (size_t r = 0; r < num_sort_ranks; r++) { segment_diff_buffers[r].destroy(); }

#ifdef DEBUG_CUNUMERIC
    for (size_t segment = 0; segment < num_segments_l; ++segment) {
      size_t start_position = segment * num_sort_ranks;
      assert(0 == thrust::reduce(exec_policy,
                                 segment_diff_2d.ptr(start_position),
                                 segment_diff_2d.ptr(start_position) + num_sort_ranks,
                                 0));
    }
#endif

    // 2d data [segments][ranks]
    /*
          -2    2    1    1    -3     2    -1
          -2    0    1    2    -1     1     0  (inclusive scan)
          neg --> receive from right
          pos --> send right

          0    2    0    -1   -2    1    -1    (incl.scan right)
          neg --> receive from left
          pos --> send left

          edge case --> send more than whole line should not happen due to sample choice!
    */
    // 2 (signed) arrays - left/right for every segment
    auto send_left  = create_buffer<int64_t>(num_segments_l, Memory::Z_COPY_MEM);
    auto send_right = create_buffer<int64_t>(num_segments_l, Memory::Z_COPY_MEM);

    // compute data to send....
    auto segment_diff_2d_scan =
      create_buffer<int64_t>(num_segments_l * num_sort_ranks, Memory::GPU_FB_MEM);
    thrust::device_ptr<int64_t> segment_diff_2d_ptr(segment_diff_2d.ptr(0));
    thrust::device_ptr<int64_t> segment_diff_2d_scan_ptr(segment_diff_2d_scan.ptr(0));
    thrust::inclusive_scan(exec_policy,
                           segment_diff_2d_ptr,
                           segment_diff_2d_ptr + num_segments_l * num_sort_ranks,
                           segment_diff_2d_scan_ptr);
    CHECK_CUDA(cudaMemcpy2DAsync(send_right.ptr(0),
                                 sizeof(int64_t),
                                 segment_diff_2d_scan.ptr(0) + my_sort_rank,
                                 num_sort_ranks * sizeof(int64_t),
                                 sizeof(int64_t),
                                 num_segments_l,
                                 cudaMemcpyDeviceToDevice,
                                 stream));
    thrust::reverse_iterator<thrust::device_vector<int64_t>::iterator> iter_in(
      segment_diff_2d_ptr + num_segments_l * num_sort_ranks);
    thrust::reverse_iterator<thrust::device_vector<int64_t>::iterator> iter_out(
      segment_diff_2d_scan_ptr + num_segments_l * num_sort_ranks);
    thrust::inclusive_scan(
      exec_policy, iter_in, iter_in + num_segments_l * num_sort_ranks, iter_out);
    CHECK_CUDA(cudaMemcpy2DAsync(send_left.ptr(0),
                                 sizeof(int64_t),
                                 segment_diff_2d_scan.ptr(0) + my_sort_rank,
                                 num_sort_ranks * sizeof(int64_t),
                                 sizeof(int64_t),
                                 num_segments_l,
                                 cudaMemcpyDeviceToDevice,
                                 stream));
    segment_diff_2d.destroy();
    segment_diff_2d_scan.destroy();

    // package data to send
    size_t send_left_size  = thrust::transform_reduce(exec_policy,
                                                     send_left.ptr(0),
                                                     send_left.ptr(0) + num_segments_l,
                                                     positive_value(),
                                                     0,
                                                     thrust::plus<int64_t>());
    size_t recv_left_size  = thrust::transform_reduce(exec_policy,
                                                     send_left.ptr(0),
                                                     send_left.ptr(0) + num_segments_l,
                                                     negative_value(),
                                                     0,
                                                     thrust::plus<int64_t>());
    size_t send_right_size = thrust::transform_reduce(exec_policy,
                                                      send_right.ptr(0),
                                                      send_right.ptr(0) + num_segments_l,
                                                      positive_value(),
                                                      0,
                                                      thrust::plus<int64_t>());
    size_t recv_right_size = thrust::transform_reduce(exec_policy,
                                                      send_right.ptr(0),
                                                      send_right.ptr(0) + num_segments_l,
                                                      negative_value(),
                                                      0,
                                                      thrust::plus<int64_t>());
    SortPiece<VAL> send_left_data, recv_left_data, send_right_data, recv_right_data;
    send_left_data.values  = create_buffer<VAL>(send_left_size, Memory::GPU_FB_MEM);
    recv_left_data.values  = create_buffer<VAL>(recv_left_size, Memory::GPU_FB_MEM);
    send_right_data.values = create_buffer<VAL>(send_right_size, Memory::GPU_FB_MEM);
    recv_right_data.values = create_buffer<VAL>(recv_right_size, Memory::GPU_FB_MEM);
    send_left_data.size    = 0;  // will be incremented when data is inserted
    recv_left_data.size    = recv_left_size;
    send_right_data.size   = 0;  // will be incremented when data is inserted
    recv_right_data.size   = recv_right_size;
    if (argsort) {
      send_left_data.indices  = create_buffer<int64_t>(send_left_size, Memory::GPU_FB_MEM);
      recv_left_data.indices  = create_buffer<int64_t>(recv_left_size, Memory::GPU_FB_MEM);
      send_right_data.indices = create_buffer<int64_t>(send_right_size, Memory::GPU_FB_MEM);
      recv_right_data.indices = create_buffer<int64_t>(recv_right_size, Memory::GPU_FB_MEM);
    }
    auto segment_diff_ptr = segment_diff.ptr(0);
    auto send_left_ptr    = send_left.ptr(0);
    auto send_right_ptr   = send_right.ptr(0);
    {
      size_t start_pos = 0;
      for (size_t segment = 0; segment < num_segments_l; ++segment) {
        size_t end_pos = start_pos + segment_size_l + segment_diff_ptr[segment];
        if (send_left_ptr[segment] > 0) {
          auto size = send_left_ptr[segment];
          CHECK_CUDA(cudaMemcpyAsync(send_left_data.values.ptr(send_left_data.size),
                                     merge_buffers[0].values.ptr(start_pos),
                                     sizeof(VAL) * size,
                                     cudaMemcpyDeviceToDevice,
                                     stream));
          if (argsort) {
            CHECK_CUDA(cudaMemcpyAsync(send_left_data.indices.ptr(send_left_data.size),
                                       merge_buffers[0].indices.ptr(start_pos),
                                       sizeof(int64_t) * size,
                                       cudaMemcpyDeviceToDevice,
                                       stream));
          }
          send_left_data.size += size;
        }
        if (send_right_ptr[segment] > 0) {
          auto size = send_right_ptr[segment];
          CHECK_CUDA(cudaMemcpyAsync(send_right_data.values.ptr(send_right_data.size),
                                     merge_buffers[0].values.ptr(end_pos - size),
                                     sizeof(VAL) * size,
                                     cudaMemcpyDeviceToDevice,
                                     stream));
          if (argsort) {
            CHECK_CUDA(cudaMemcpyAsync(send_right_data.indices.ptr(send_right_data.size),
                                       merge_buffers[0].indices.ptr(end_pos - size),
                                       sizeof(int64_t) * size,
                                       cudaMemcpyDeviceToDevice,
                                       stream));
          }
          send_right_data.size += size;
        }
        start_pos = end_pos;
      }
      assert(send_left_data.size == send_left_size);
      assert(send_right_data.size == send_right_size);
    }

    // send/receive overlapping data
    {
      CHECK_NCCL(ncclGroupStart());
      if (send_left_size > 0) {
        CHECK_NCCL(ncclSend(send_left_data.values.ptr(0),
                            send_left_data.size * sizeof(VAL),
                            ncclInt8,
                            sort_ranks[my_sort_rank - 1],
                            *comm,
                            stream));
      }
      if (send_right_size > 0) {
        CHECK_NCCL(ncclSend(send_right_data.values.ptr(0),
                            send_right_data.size * sizeof(VAL),
                            ncclInt8,
                            sort_ranks[my_sort_rank + 1],
                            *comm,
                            stream));
      }
      if (recv_left_size > 0) {
        CHECK_NCCL(ncclRecv(recv_left_data.values.ptr(0),
                            recv_left_data.size * sizeof(VAL),
                            ncclInt8,
                            sort_ranks[my_sort_rank - 1],
                            *comm,
                            stream));
      }
      if (recv_right_size > 0) {
        CHECK_NCCL(ncclRecv(recv_right_data.values.ptr(0),
                            recv_right_data.size * sizeof(VAL),
                            ncclInt8,
                            sort_ranks[my_sort_rank + 1],
                            *comm,
                            stream));
      }

      if (argsort) {
        if (send_left_size > 0) {
          CHECK_NCCL(ncclSend(send_left_data.indices.ptr(0),
                              send_left_data.size,
                              ncclInt64,
                              sort_ranks[my_sort_rank - 1],
                              *comm,
                              stream));
        }
        if (send_right_size > 0) {
          CHECK_NCCL(ncclSend(send_right_data.indices.ptr(0),
                              send_right_data.size,
                              ncclInt64,
                              sort_ranks[my_sort_rank + 1],
                              *comm,
                              stream));
        }
        if (recv_left_size > 0) {
          CHECK_NCCL(ncclRecv(recv_left_data.indices.ptr(0),
                              recv_left_data.size,
                              ncclInt64,
                              sort_ranks[my_sort_rank - 1],
                              *comm,
                              stream));
        }
        if (recv_right_size > 0) {
          CHECK_NCCL(ncclRecv(recv_right_data.indices.ptr(0),
                              recv_right_data.size,
                              ncclInt64,
                              sort_ranks[my_sort_rank + 1],
                              *comm,
                              stream));
        }
      }
      CHECK_NCCL(ncclGroupEnd());

      send_left_data.values.destroy();
      send_right_data.values.destroy();
      if (argsort) {
        send_left_data.indices.destroy();
        send_right_data.indices.destroy();
      }
    }

    // merge data into target
    result.size   = volume;
    result.values = create_buffer<VAL>(volume, Memory::GPU_FB_MEM);
    if (argsort) { result.indices = create_buffer<int64_t>(volume, Memory::GPU_FB_MEM); }

    {
      size_t start_pos      = 0;
      size_t result_pos     = 0;
      size_t left_read_pos  = 0;
      size_t right_read_pos = 0;
      for (size_t segment = 0; segment < num_segments_l; ++segment) {
        size_t end_pos = start_pos + segment_size_l + segment_diff_ptr[segment];

        size_t copy_start = start_pos;
        size_t copy_end   = end_pos;

        if (send_left_ptr[segment] < 0) {
          // we have data to merge
          size_t received_size = -send_left_ptr[segment];
          CHECK_CUDA(cudaMemcpyAsync(result.values.ptr(result_pos),
                                     recv_left_data.values.ptr(left_read_pos),
                                     sizeof(VAL) * received_size,
                                     cudaMemcpyDeviceToDevice,
                                     stream));
          if (argsort)
            CHECK_CUDA(cudaMemcpyAsync(result.indices.ptr(result_pos),
                                       recv_left_data.indices.ptr(left_read_pos),
                                       sizeof(int64_t) * received_size,
                                       cudaMemcpyDeviceToDevice,
                                       stream));
          result_pos += received_size;
          left_read_pos += received_size;
        }

        // assemble line from old data and received data
        if (send_left_ptr[segment] > 0) copy_start += send_left_ptr[segment];
        if (send_right_ptr[segment] > 0) copy_end -= send_right_ptr[segment];
        {
          CHECK_CUDA(cudaMemcpyAsync(result.values.ptr(result_pos),
                                     merge_buffers[0].values.ptr(copy_start),
                                     sizeof(VAL) * (copy_end - copy_start),
                                     cudaMemcpyDeviceToDevice,
                                     stream));
          if (argsort)
            CHECK_CUDA(cudaMemcpyAsync(result.indices.ptr(result_pos),
                                       merge_buffers[0].indices.ptr(copy_start),
                                       sizeof(int64_t) * (copy_end - copy_start),
                                       cudaMemcpyDeviceToDevice,
                                       stream));
          result_pos += (copy_end - copy_start);
        }

        if (send_right_ptr[segment] < 0) {
          // we have data to merge
          size_t received_size = -send_right_ptr[segment];
          CHECK_CUDA(cudaMemcpyAsync(result.values.ptr(result_pos),
                                     recv_right_data.values.ptr(right_read_pos),
                                     sizeof(VAL) * received_size,
                                     cudaMemcpyDeviceToDevice,
                                     stream));
          if (argsort)
            CHECK_CUDA(cudaMemcpyAsync(result.indices.ptr(result_pos),
                                       recv_right_data.indices.ptr(right_read_pos),
                                       sizeof(int64_t) * received_size,
                                       cudaMemcpyDeviceToDevice,
                                       stream));
          result_pos += received_size;
          right_read_pos += received_size;
        }

        assert(result_pos == (segment + 1) * segment_size_l);
        start_pos = end_pos;
      }
    }

    // remove segment_sizes, all buffers should be destroyed...
    segment_diff.destroy();
    send_left.destroy();
    send_right.destroy();
    merge_buffers[0].values.destroy();
    recv_left_data.values.destroy();
    recv_right_data.values.destroy();
    if (argsort) {
      merge_buffers[0].indices.destroy();
      recv_left_data.indices.destroy();
      recv_right_data.indices.destroy();
    }
  }

  return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template <LegateTypeCode CODE, int32_t DIM>
struct SortImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(const Array& input_array,
                  Array& output_array,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t volume,
                  const size_t segment_size_l,
                  const size_t segment_size_g,
                  const bool argsort,
                  const bool stable,
                  const bool is_index_space,
                  const size_t local_rank,
                  const size_t num_ranks,
                  const size_t num_sort_ranks,
                  const std::vector<comm::Communicator>& comms)
  {
    auto input = input_array.read_accessor<VAL, DIM>(rect);

    // we allow empty domains for distributed sorting
    assert(rect.empty() || input.accessor.is_dense_row_major(rect));

    auto stream                = get_cached_stream();
    bool need_distributed_sort = segment_size_l != segment_size_g;

    // initialize sort pointers
    SortPiece<VAL> local_sorted;
    int64_t* indices_ptr = nullptr;
    VAL* values_ptr      = nullptr;
    if (argsort) {
      // make a buffer for input
      auto input_copy     = create_buffer<VAL>(volume, Legion::Memory::Kind::GPU_FB_MEM);
      local_sorted.values = input_copy;
      values_ptr          = input_copy.ptr(0);

      // initialize indices
      if (need_distributed_sort || output_array.dim() == -1) {
        auto indices_buffer  = create_buffer<int64_t>(volume, Legion::Memory::Kind::GPU_FB_MEM);
        indices_ptr          = indices_buffer.ptr(0);
        local_sorted.indices = indices_buffer;
        local_sorted.size    = volume;
      } else {
        AccessorWO<int64_t, DIM> output = output_array.write_accessor<int64_t, DIM>(rect);
        assert(output.accessor.is_dense_row_major(rect));
        indices_ptr = output.ptr(rect.lo);
      }
      size_t offset = rect.lo[DIM - 1];
      if (volume > 0) {
        if (DIM == 1) {
          thrust::sequence(thrust::cuda::par.on(stream), indices_ptr, indices_ptr + volume, offset);
        } else {
          thrust::transform(thrust::cuda::par.on(stream),
                            thrust::make_counting_iterator<int64_t>(0),
                            thrust::make_counting_iterator<int64_t>(volume),
                            thrust::make_constant_iterator<int64_t>(segment_size_l),
                            indices_ptr,
                            modulusWithOffset(offset));
        }
      }
    } else {
      // initialize output
      if (need_distributed_sort || output_array.dim() == -1) {
        auto input_copy      = create_buffer<VAL>(volume, Legion::Memory::Kind::GPU_FB_MEM);
        values_ptr           = input_copy.ptr(0);
        local_sorted.values  = input_copy;
        local_sorted.indices = create_buffer<int64_t>(0, Legion::Memory::Kind::GPU_FB_MEM);
        local_sorted.size    = volume;
      } else {
        AccessorWO<VAL, DIM> output = output_array.write_accessor<VAL, DIM>(rect);
        assert(output.accessor.is_dense_row_major(rect));
        values_ptr = output.ptr(rect.lo);
      }
    }
    if (volume > 0) {
      // sort data (locally)
      local_sort<CODE>(input.ptr(rect.lo),
                       values_ptr,
                       indices_ptr,
                       indices_ptr,
                       volume,
                       segment_size_l,
                       stable,
                       stream);
    }

    // this is linked to the decision in sorting.py on when to use an 'unbounded' output array.
    if (output_array.dim() == -1) {
      assert(DIM == 1);
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
      if (need_distributed_sort) {
        assert(DIM > 1);
        assert(is_index_space);
        std::vector<size_t> sort_ranks(num_sort_ranks);
        size_t rank_group = local_rank / num_sort_ranks;
        for (int r = 0; r < num_sort_ranks; ++r) sort_ranks[r] = rank_group * num_sort_ranks + r;
        SortPiece<VAL> final_sorted_flattened = sample_sort_nccl_nd(local_sorted,
                                                                    local_rank,
                                                                    num_ranks,
                                                                    segment_size_g,
                                                                    local_rank % num_sort_ranks,
                                                                    num_sort_ranks,
                                                                    &sort_ranks[0],
                                                                    segment_size_l,
                                                                    argsort,
                                                                    stream,
                                                                    comms[0].get<ncclComm_t*>());
        assert(final_sorted_flattened.size == volume);

        if (argsort) {
          auto output = output_array.write_accessor<int64_t, DIM>(rect);
          CHECK_CUDA(cudaMemcpyAsync(output.ptr(rect.lo),
                                     final_sorted_flattened.indices.ptr(0),
                                     sizeof(int64_t) * volume,
                                     cudaMemcpyDeviceToDevice,
                                     stream));
          final_sorted_flattened.indices.destroy();
        } else {
          auto output = output_array.write_accessor<VAL, DIM>(rect);
          CHECK_CUDA(cudaMemcpyAsync(output.ptr(rect.lo),
                                     final_sorted_flattened.values.ptr(0),
                                     sizeof(VAL) * volume,
                                     cudaMemcpyDeviceToDevice,
                                     stream));
        }
        final_sorted_flattened.values.destroy();
      } else if (argsort) {
        // cleanup
        local_sorted.values.destroy();
      }
    }
  }
};

/*static*/ void SortTask::gpu_variant(TaskContext& context)
{
  sort_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
