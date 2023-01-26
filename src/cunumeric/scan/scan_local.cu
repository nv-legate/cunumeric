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

#include "cunumeric/scan/scan_local.h"
#include "cunumeric/scan/scan_local_template.inl"
#include "cunumeric/unary/isnan.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <typename RES>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  lazy_kernel(RES* out, RES* sum_val)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 1) return;
  sum_val[0] = out[0];
}

template <typename RES, int DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) partition_sum(
  RES* out, Buffer<RES, DIM> sum_val, const Pitches<DIM - 1> pitches, uint64_t len, uint64_t stride)
{
  unsigned int tid = threadIdx.x;
  uint64_t blid    = blockIdx.x * blockDim.x;

  uint64_t index = (blid + tid) * stride;

  if (index < len) {
    auto sum_valp     = pitches.unflatten(index, Point<DIM>::ZEROES());
    sum_valp[DIM - 1] = 0;
    sum_val[sum_valp] = out[index + stride - 1];
  }
}

template <typename RES, typename OP>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  cuda_add(RES* B, uint64_t len, uint64_t stride, OP func, RES* block_sum)
{
  unsigned int tid  = threadIdx.x;
  unsigned int blid = blockIdx.x * blockDim.x;

  uint64_t pad_stride = stride;
  bool must_copy      = true;
  if (stride & (stride - 1)) {
    pad_stride = 1 << (32 - __clz(stride));
    must_copy  = (tid & (pad_stride - 1)) < stride;
  }
  uint64_t blocks_per_batch;
  bool last_block;
  bool first_block;

  blocks_per_batch    = (stride - 1) / THREADS_PER_BLOCK + 1;
  pad_stride          = blocks_per_batch * THREADS_PER_BLOCK;
  last_block          = (blockIdx.x + 1) % blocks_per_batch == 0;
  first_block         = (blockIdx.x) % blocks_per_batch == 0;
  int remaining_batch = stride % THREADS_PER_BLOCK;
  if (remaining_batch == 0) { remaining_batch = THREADS_PER_BLOCK; }
  must_copy = !last_block || (tid < remaining_batch);

  int pad_per_batch = pad_stride - stride;

  uint64_t idx0 = tid + blid;

  uint64_t batch_id = idx0 / pad_stride;
  idx0              = idx0 - pad_per_batch * batch_id;

  if (idx0 < len && must_copy && !first_block) {
    B[idx0] = func(block_sum[blockIdx.x - 1], B[idx0]);
  }
}

template <typename RES, typename OP>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) batch_scan_cuda(
  const RES* A, RES* B, uint64_t len, uint64_t stride, OP func, RES identity, RES* block_sum)
{
  __shared__ RES temp[THREADS_PER_BLOCK];

  unsigned int tid  = threadIdx.x;
  unsigned int blid = blockIdx.x * blockDim.x;

  uint64_t pad_stride = stride;
  bool must_copy      = true;
  if (stride & (stride - 1)) {
    pad_stride = 1 << (32 - __clz(stride));
    must_copy  = (tid & (pad_stride - 1)) < stride;
  }
  bool last_block;
  if (pad_stride > THREADS_PER_BLOCK) {
    uint64_t blocks_per_batch = (stride - 1) / THREADS_PER_BLOCK + 1;
    pad_stride                = blocks_per_batch * THREADS_PER_BLOCK;
    last_block                = (blockIdx.x + 1) % blocks_per_batch == 0;
    int remaining_batch       = stride % THREADS_PER_BLOCK;
    if (remaining_batch == 0) { remaining_batch = THREADS_PER_BLOCK; }
    must_copy = !last_block || (tid < remaining_batch);
  }

  int pad_per_batch   = pad_stride - stride;
  int n_batches_block = THREADS_PER_BLOCK / pad_stride;

  uint64_t idx0 = tid + blid;

  uint64_t batch_id = idx0 / pad_stride;
  idx0              = idx0 - pad_per_batch * batch_id;

  if (idx0 < len) {
    temp[tid] = (must_copy) ? A[idx0] : identity;
    __syncthreads();
    if (!n_batches_block) {
      n_batches_block = 1;
      pad_stride      = THREADS_PER_BLOCK;
    }
    for (int j = 0; j < n_batches_block; j++) {
      int offset = j * pad_stride;
      for (int i = 1; i <= pad_stride; i <<= 1) {
        int index       = ((tid + 1) * 2 * i - 1);
        int index_block = offset + index;
        if (index < (pad_stride)) {
          temp[index_block] = func(temp[index_block - i], temp[index_block]);
        }
        __syncthreads();
      }
      for (int i = pad_stride >> 1; i > 0; i >>= 1) {
        int index       = ((tid + 1) * 2 * i - 1);
        int index_block = offset + index;
        if ((index + i) < (pad_stride)) {
          temp[index_block + i] = func(temp[index_block], temp[index_block + i]);
        }
        __syncthreads();
      }
    }
    if (must_copy) { B[idx0] = temp[tid]; }
    if (block_sum != nullptr && tid == THREADS_PER_BLOCK - 1 && !last_block) {
      block_sum[blockIdx.x] = temp[tid];
    }
  }
}

template <typename RES, typename OP>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) batch_scan_cuda_nan(
  const RES* A, RES* B, uint64_t len, uint64_t stride, OP func, RES identity, RES* block_sum)
{
  __shared__ RES temp[THREADS_PER_BLOCK];

  unsigned int tid  = threadIdx.x;
  unsigned int blid = blockIdx.x * blockDim.x;

  uint64_t pad_stride = stride;
  bool must_copy      = true;
  if (stride & (stride - 1)) {
    pad_stride = 1 << (32 - __clz(stride));
    must_copy  = (tid & (pad_stride - 1)) < stride;
  }
  bool last_block;
  if (pad_stride > THREADS_PER_BLOCK) {
    uint64_t blocks_per_batch = (stride - 1) / THREADS_PER_BLOCK + 1;
    pad_stride                = blocks_per_batch * THREADS_PER_BLOCK;
    last_block                = (blockIdx.x + 1) % blocks_per_batch == 0;
    int remaining_batch       = stride % THREADS_PER_BLOCK;
    if (remaining_batch == 0) { remaining_batch = THREADS_PER_BLOCK; }
    must_copy = !last_block || (tid < remaining_batch);
  }

  int pad_per_batch   = pad_stride - stride;
  int n_batches_block = THREADS_PER_BLOCK / pad_stride;

  uint64_t idx0 = tid + blid;

  uint64_t batch_id = idx0 / pad_stride;
  idx0              = idx0 - pad_per_batch * batch_id;

  if (idx0 < len) {
    RES val   = (must_copy) ? A[idx0] : identity;
    temp[tid] = cunumeric::is_nan(val) ? identity : val;
    __syncthreads();
    if (!n_batches_block) {
      n_batches_block = 1;
      pad_stride      = THREADS_PER_BLOCK;
    }
    for (int j = 0; j < n_batches_block; j++) {
      int offset = j * pad_stride;
      for (int i = 1; i <= pad_stride; i <<= 1) {
        int index       = ((tid + 1) * 2 * i - 1);
        int index_block = offset + index;
        if (index < (pad_stride)) {
          temp[index_block] = func(temp[index_block - i], temp[index_block]);
        }
        __syncthreads();
      }
      for (int i = pad_stride >> 1; i > 0; i >>= 1) {
        int index       = ((tid + 1) * 2 * i - 1);
        int index_block = offset + index;
        if ((index + i) < (pad_stride)) {
          temp[index_block + i] = func(temp[index_block], temp[index_block + i]);
        }
        __syncthreads();
      }
    }
    if (must_copy) { B[idx0] = temp[tid]; }
    if (block_sum != nullptr && tid == THREADS_PER_BLOCK - 1 && !last_block) {
      block_sum[blockIdx.x] = temp[tid];
    }
  }
}

template <typename RES, typename OP>
void cuda_scan(
  const RES* A, RES* B, uint64_t len, uint64_t stride, OP func, RES identity, cudaStream_t stream)
{
  assert(stride != 0);
  uint64_t pad_stride = 1 << (32 - __builtin_clz(stride));
  if (pad_stride > THREADS_PER_BLOCK) {
    uint64_t blocks_per_batch = (stride - 1) / THREADS_PER_BLOCK + 1;
    pad_stride                = blocks_per_batch * THREADS_PER_BLOCK;
  }
  uint64_t pad_len  = (len / stride) * pad_stride;
  uint64_t grid_dim = (pad_len - 1) / THREADS_PER_BLOCK + 1;

  RES* blocked_sum = nullptr;
  uint64_t blocked_len, blocked_stride;
  if (stride > THREADS_PER_BLOCK) {
    blocked_len    = grid_dim;
    blocked_stride = grid_dim / (len / stride);
    CHECK_CUDA(cudaMalloc(&blocked_sum, blocked_len * sizeof(RES)));
  }

  batch_scan_cuda<RES, OP>
    <<<grid_dim, THREADS_PER_BLOCK, 0, stream>>>(A, B, len, stride, func, identity, blocked_sum);
  CHECK_CUDA_STREAM(stream);

  if (stride > THREADS_PER_BLOCK) {
    cuda_scan(blocked_sum, blocked_sum, blocked_len, blocked_stride, func, identity, stream);
    cuda_add<<<grid_dim, THREADS_PER_BLOCK, 0, stream>>>(B, len, stride, func, blocked_sum);
    CHECK_CUDA_STREAM(stream);
  }

  if (stride > THREADS_PER_BLOCK) { CHECK_CUDA(cudaFree(blocked_sum)); }
}

template <typename RES, typename OP>
void cuda_scan_nan(
  const RES* A, RES* B, uint64_t len, uint64_t stride, OP func, RES identity, cudaStream_t stream)
{
  assert(stride != 0);
  uint64_t pad_stride = 1 << (32 - __builtin_clz(stride));
  if (pad_stride > THREADS_PER_BLOCK) {
    uint64_t blocks_per_batch = (stride - 1) / THREADS_PER_BLOCK + 1;
    pad_stride                = blocks_per_batch * THREADS_PER_BLOCK;
  }
  uint64_t pad_len  = (len / stride) * pad_stride;
  uint64_t grid_dim = (pad_len - 1) / THREADS_PER_BLOCK + 1;

  RES* blocked_sum = nullptr;
  uint64_t blocked_len, blocked_stride;
  if (stride > THREADS_PER_BLOCK) {
    blocked_len    = grid_dim;
    blocked_stride = grid_dim / (len / stride);
    CHECK_CUDA(cudaMalloc(&blocked_sum, blocked_len * sizeof(RES)));
  }

  batch_scan_cuda_nan<RES, OP>
    <<<grid_dim, THREADS_PER_BLOCK, 0, stream>>>(A, B, len, stride, func, identity, blocked_sum);
  CHECK_CUDA_STREAM(stream);

  if (stride > THREADS_PER_BLOCK) {
    cuda_scan(blocked_sum, blocked_sum, blocked_len, blocked_stride, func, identity, stream);
    cuda_add<<<grid_dim, THREADS_PER_BLOCK, 0, stream>>>(B, len, stride, func, blocked_sum);
    CHECK_CUDA_STREAM(stream);
  }

  if (stride > THREADS_PER_BLOCK) { CHECK_CUDA(cudaFree(blocked_sum)); }
}

template <ScanCode OP_CODE, LegateTypeCode CODE, int DIM>
struct ScanLocalImplBody<VariantKind::GPU, OP_CODE, CODE, DIM> {
  using OP  = ScanOp<OP_CODE, CODE>;
  using VAL = legate_type_of<CODE>;

  void operator()(OP func,
                  AccessorWO<VAL, DIM>& out,
                  const AccessorRO<VAL, DIM>& in,
                  Array& sum_vals,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect) const
  {
    auto outptr = out.ptr(rect.lo);
    auto inptr  = in.ptr(rect.lo);
    auto volume = rect.volume();

    auto stride = rect.hi[DIM - 1] - rect.lo[DIM - 1] + 1;

    auto stream = get_cached_stream();

    Point<DIM> extents = rect.hi - rect.lo + Point<DIM>::ONES();
    extents[DIM - 1]   = 1;  // one element along scan axis

    auto sum_valsptr = sum_vals.create_output_buffer<VAL, DIM>(extents, true);

    VAL identity = (VAL)ScanOp<OP_CODE, CODE>::nan_identity;

    if (volume == stride) {
      // Thrust is slightly faster for the 1D case
      thrust::inclusive_scan(thrust::cuda::par.on(stream), inptr, inptr + stride, outptr, func);
    } else {
      cuda_scan<VAL, OP>(inptr, outptr, volume, stride, func, identity, stream);
    }

    uint64_t grid_dim = ((volume / stride) - 1) / THREADS_PER_BLOCK + 1;
    partition_sum<<<grid_dim, THREADS_PER_BLOCK, 0, stream>>>(
      outptr, sum_valsptr, pitches, volume, stride);
    CHECK_CUDA_STREAM(stream);
  }
};

template <ScanCode OP_CODE, LegateTypeCode CODE, int DIM>
struct ScanLocalNanImplBody<VariantKind::GPU, OP_CODE, CODE, DIM> {
  using OP  = ScanOp<OP_CODE, CODE>;
  using VAL = legate_type_of<CODE>;

  struct convert_nan_func {
    __device__ VAL operator()(VAL x)
    {
      return cunumeric::is_nan(x) ? (VAL)ScanOp<OP_CODE, CODE>::nan_identity : x;
    }
  };

  void operator()(OP func,
                  AccessorWO<VAL, DIM>& out,
                  const AccessorRO<VAL, DIM>& in,
                  Array& sum_vals,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect) const
  {
    auto outptr = out.ptr(rect.lo);
    auto inptr  = in.ptr(rect.lo);
    auto volume = rect.volume();

    auto stride = rect.hi[DIM - 1] - rect.lo[DIM - 1] + 1;

    auto stream = get_cached_stream();

    Point<DIM> extents = rect.hi - rect.lo + Point<DIM>::ONES();
    extents[DIM - 1]   = 1;  // one element along scan axis

    auto sum_valsptr = sum_vals.create_output_buffer<VAL, DIM>(extents, true);

    VAL identity = (VAL)ScanOp<OP_CODE, CODE>::nan_identity;

    if (volume == stride) {
      // Thrust is slightly faster for the 1D case
      thrust::inclusive_scan(thrust::cuda::par.on(stream),
                             thrust::make_transform_iterator(inptr, convert_nan_func()),
                             thrust::make_transform_iterator(inptr + stride, convert_nan_func()),
                             outptr,
                             func);
    } else {
      cuda_scan_nan<VAL, OP>(inptr, outptr, volume, stride, func, identity, stream);
    }

    uint64_t grid_dim = ((volume / stride) - 1) / THREADS_PER_BLOCK + 1;
    partition_sum<<<grid_dim, THREADS_PER_BLOCK, 0, stream>>>(
      outptr, sum_valsptr, pitches, volume, stride);
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void ScanLocalTask::gpu_variant(TaskContext& context)
{
  scan_local_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
