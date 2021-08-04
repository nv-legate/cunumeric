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

#include "unary/unary_op.h"
#include "unary/unary_op_template.inl"

namespace legate {
namespace numpy {

using namespace Legion;

template <typename Function, typename ARG, typename RES>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  dense_kernel(size_t volume, Function func, RES* out, const ARG* in)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  out[idx] = func(in[idx]);
}

template <typename Function, typename ReadAcc, typename WriteAcc, typename Pitches, typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  generic_kernel(size_t volume, Function func, WriteAcc out, ReadAcc in, Pitches pitches, Rect rect)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto point = pitches.unflatten(idx, rect.lo);
  out[point] = func(in[point]);
}

template <UnaryOpCode OP_CODE, LegateTypeCode CODE, int DIM>
struct UnaryOpImplBody<VariantKind::GPU, OP_CODE, CODE, DIM> {
  using OP  = UnaryOp<OP_CODE, CODE>;
  using ARG = typename OP::T;
  using RES = std::result_of_t<OP(ARG)>;

  void operator()(OP func,
                  AccessorWO<RES, DIM> out,
                  AccessorRO<ARG, DIM> in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dense) {
      auto outptr = out.ptr(rect);
      auto inptr  = in.ptr(rect);
      dense_kernel<<<blocks, THREADS_PER_BLOCK>>>(volume, func, outptr, inptr);
    } else {
      generic_kernel<<<blocks, THREADS_PER_BLOCK>>>(volume, func, out, in, pitches, rect);
    }
  }
};

/*static*/ void UnaryOpTask::gpu_variant(TaskContext& context)
{
  unary_op_template<VariantKind::GPU>(context);
}

}  // namespace numpy
}  // namespace legate
