/* Copyright 2021 NVIDIA Corporation
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

#include "unary/scalar_unary_red.h"
#include "unary/scalar_unary_red_template.inl"

#include "cuda_help.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <typename Op,
          typename Output,
          typename ReadAcc,
          typename Pitches,
          typename Point,
          typename VAL>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  reduction_kernel(size_t volume,
                   Op op,
                   Output out,
                   ReadAcc in,
                   Pitches pitches,
                   Point origin,
                   size_t iters,
                   VAL identity)
{
  auto value = identity;
  for (size_t idx = 0; idx < iters; idx++) {
    const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < volume) {
      auto point = pitches.unflatten(offset, origin);
      Op::template fold<true>(value, in[point]);
    }
  }
  // Every thread in the thread block must participate in the exchange to get correct results
  reduce_output(out, value);
}

template <typename Output, typename ReadAcc, typename Pitches, typename Point, typename VAL>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) contains_kernel(
  size_t volume, Output out, ReadAcc in, Pitches pitches, Point origin, size_t iters, VAL to_find)
{
  bool value = false;
  for (size_t idx = 0; idx < iters; idx++) {
    const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < volume) {
      auto point = pitches.unflatten(offset, origin);
      SumReduction<bool>::fold<true>(value, in[point] == to_find);
    }
  }
  // Every thread in the thread block must participate in the exchange to get correct results
  reduce_output(out, value);
}

template <typename Output, typename Pitches, typename Point, typename VAL, int32_t DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) count_nonzero_kernel(
  size_t volume, Output out, AccessorRO<VAL, DIM> in, Pitches pitches, Point origin, size_t iters)
{
  uint64_t value = 0;
  for (size_t idx = 0; idx < iters; idx++) {
    const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < volume) {
      auto point = pitches.unflatten(offset, origin);
      SumReduction<uint64_t>::fold<true>(value, in[point] != VAL(0));
    }
  }
  // Every thread in the thread block must participate in the exchange to get correct results
  reduce_output(out, value);
}

template <UnaryRedCode OP_CODE, LegateTypeCode CODE, int DIM>
struct ScalarUnaryRedImplBody<VariantKind::GPU, OP_CODE, CODE, DIM> {
  using OP    = UnaryRedOp<OP_CODE, CODE>;
  using LG_OP = typename OP::OP;
  using VAL   = legate_type_of<CODE>;

  void operator()(OP func,
                  VAL& result,
                  AccessorRO<VAL, DIM> in,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  bool dense) const
  {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const size_t volume = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    DeferredReduction<typename OP::OP> out;
    size_t shmem_size = THREADS_PER_BLOCK / 32 * sizeof(VAL);

    if (blocks >= MAX_REDUCTION_CTAS) {
      const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
      reduction_kernel<<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, typename OP::OP{}, out, in, pitches, rect.lo, iters, LG_OP::identity);
    } else
      reduction_kernel<<<blocks, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, typename OP::OP{}, out, in, pitches, rect.lo, 1, LG_OP::identity);

    // TODO: We eventually want to unblock this step
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    result = out.read();
  }
};

template <LegateTypeCode CODE, int DIM>
struct ScalarUnaryRedImplBody<VariantKind::GPU, UnaryRedCode::CONTAINS, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(bool& result,
                  AccessorRO<VAL, DIM> in,
                  const UntypedScalar& to_find_scalar,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  bool dense) const
  {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const auto to_find  = to_find_scalar.value<VAL>();
    const size_t volume = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    DeferredReduction<SumReduction<bool>> out;
    size_t shmem_size = THREADS_PER_BLOCK / 32 * sizeof(bool);

    if (blocks >= MAX_REDUCTION_CTAS) {
      const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
      contains_kernel<<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, out, in, pitches, rect.lo, iters, to_find);
    } else
      contains_kernel<<<blocks, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, out, in, pitches, rect.lo, 1, to_find);

    // TODO: We eventually want to unblock this step
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    result = out.read();
  }
};

template <LegateTypeCode CODE, int DIM>
struct ScalarUnaryRedImplBody<VariantKind::GPU, UnaryRedCode::COUNT_NONZERO, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(uint64_t& result,
                  AccessorRO<VAL, DIM> in,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  bool dense) const
  {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const size_t volume = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    DeferredReduction<SumReduction<uint64_t>> out;
    size_t shmem_size = THREADS_PER_BLOCK / 32 * sizeof(uint64_t);

    if (blocks >= MAX_REDUCTION_CTAS) {
      const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
      count_nonzero_kernel<<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, out, in, pitches, rect.lo, iters);
    } else
      count_nonzero_kernel<<<blocks, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, out, in, pitches, rect.lo, 1);

    // TODO: We eventually want to unblock this step
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    result = out.read();
  }
};

/*static*/ UntypedScalar ScalarUnaryRedTask::gpu_variant(TaskContext& context)
{
  return scalar_unary_red_template<VariantKind::GPU>(context);
}

}  // namespace numpy
}  // namespace legate
