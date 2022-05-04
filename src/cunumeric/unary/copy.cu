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

#include "cunumeric/unary/copy.h"
#include "cunumeric/unary/copy_template.inl"
#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <typename IN, typename RES>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  dense_kernel(size_t volume, RES* out, const IN* in)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  out[idx] = in[idx];
}

template <typename ReadAcc, typename WriteAcc, typename Pitches, typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  generic_kernel(size_t volume, WriteAcc out, ReadAcc in, Pitches pitches, Rect rect)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto point = pitches.unflatten(idx, rect.lo);
  out[point] = in[point];
}

template <int DIM, typename VAL>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) indirect_dense_kernel(
  size_t volume, VAL* out, const AccessorRO<VAL, DIM> in, const Point<DIM>* indirect)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  out[idx] = in[indirect[idx]];
}

template <int DIM, typename VAL>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) indirect_dense_kernel(
  size_t volume, AccessorWO<VAL, DIM> out, const VAL* in, const Point<DIM>* indirect)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  out[indirect[idx]] = in[idx];
}

template <typename ReadAcc,
          typename WriteAcc,
          typename IndirectAcc,
          typename Pitches,
          typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  indirect_generic_kernel(size_t volume,
                          WriteAcc out,
                          ReadAcc in,
                          IndirectAcc indirect,
                          Pitches pitches,
                          Rect rect,
                          bool is_source_indirect)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto point = pitches.unflatten(idx, rect.lo);
  if (is_source_indirect)
    out[point] = in[indirect[point]];
  else
    out[indirect[point]] = in[point];
}

template <LegateTypeCode CODE, int DIM>
struct CopyImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL   = legate_type_of<CODE>;
  using POINT = Point<DIM>;

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, DIM> in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto stream         = get_cached_stream();
    if (dense) {
      auto outptr = out.ptr(rect);
      auto inptr  = in.ptr(rect);
      dense_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(volume, outptr, inptr);
    } else {
      generic_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(volume, out, in, pitches, rect);
    }
    CHECK_CUDA_STREAM(stream);
  }

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, DIM> in,
                  AccessorRO<POINT, DIM> indirection,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense,
                  bool is_source_indirect) const
  {
    const size_t volume = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto stream         = get_cached_stream();
    if (dense) {
      // for the dense case all rects should be the same
      auto outptr   = out.ptr(rect);
      auto inptr    = in.ptr(rect);
      auto indirptr = indirection.ptr(rect);
      if (is_source_indirect)
        indirect_dense_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          volume, outptr, in, indirptr);
      else
        indirect_dense_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          volume, out, inptr, indirptr);
    } else {
      indirect_generic_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        volume, out, in, indirection, pitches, rect, is_source_indirect);
    }
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void CopyTask::gpu_variant(TaskContext& context)
{
  copy_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
