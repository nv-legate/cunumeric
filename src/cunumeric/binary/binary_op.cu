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

#include "cunumeric/binary/binary_op.h"
#include "cunumeric/binary/binary_op_template.inl"

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;

template <typename Function, typename RES, typename ARG>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  dense_kernel(size_t volume, Function func, RES* out, const ARG* in1, const ARG* in2)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  out[idx] = func(in1[idx], in2[idx]);
}

template <typename Function, typename WriteAcc, typename ReadAcc, typename Pitches, typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) generic_kernel(
  size_t volume, Function func, WriteAcc out, ReadAcc in1, ReadAcc in2, Pitches pitches, Rect rect)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto point = pitches.unflatten(idx, rect.lo);
  out[point] = func(in1[point], in2[point]);
}

template <BinaryOpCode OP_CODE, LegateTypeCode CODE, int DIM>
struct BinaryOpImplBody<VariantKind::GPU, OP_CODE, CODE, DIM> {
  using OP  = BinaryOp<OP_CODE, CODE>;
  using ARG = legate_type_of<CODE>;
  using RES = std::result_of_t<OP(ARG, ARG)>;

  void operator()(OP func,
                  AccessorWO<RES, DIM> out,
                  AccessorRO<ARG, DIM> in1,
                  AccessorRO<ARG, DIM> in2,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    size_t volume       = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dense) {
      auto outptr = out.ptr(rect);
      auto in1ptr = in1.ptr(rect);
      auto in2ptr = in2.ptr(rect);
      dense_kernel<<<blocks, THREADS_PER_BLOCK>>>(volume, func, outptr, in1ptr, in2ptr);
    } else {
      generic_kernel<<<blocks, THREADS_PER_BLOCK>>>(volume, func, out, in1, in2, pitches, rect);
    }
  }
};

/*static*/ void BinaryOpTask::gpu_variant(TaskContext& context)
{
  binary_op_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
