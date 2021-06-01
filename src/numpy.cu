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

#include "numpy.h"
#include "arg.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <typename T>
class CUDAReductionOpWrapper : public T {
 public:
  static const bool has_cuda_reductions = true;

  template <bool EXCLUSIVE>
  __device__ static void apply_cuda(typename T::LHS& lhs, typename T::RHS rhs)
  {
    T::template apply<EXCLUSIVE>(lhs, rhs);
  }

  template <bool EXCLUSIVE>
  __device__ static void fold_cuda(typename T::LHS& lhs, typename T::RHS rhs)
  {
    T::template fold<EXCLUSIVE>(lhs, rhs);
  }
};

#define _REGISTER_CUDA_REDOP(ID, TYPE)                                              \
  Runtime::register_reduction_op(                                                   \
    ID,                                                                             \
    Realm::ReductionOpUntyped::create_reduction_op<CUDAReductionOpWrapper<TYPE>>(), \
    NULL,                                                                           \
    NULL,                                                                           \
    false);

#define REGISTER_CUDA_REDOPS(OFFSET, OP)                                            \
  {                                                                                 \
    _REGISTER_CUDA_REDOP(OFFSET + OP<float>::REDOP_ID, OP<float>)                   \
    _REGISTER_CUDA_REDOP(OFFSET + OP<double>::REDOP_ID, OP<double>)                 \
    _REGISTER_CUDA_REDOP(OFFSET + OP<int8_t>::REDOP_ID, OP<int8_t>)                 \
    _REGISTER_CUDA_REDOP(OFFSET + OP<int16_t>::REDOP_ID, OP<int16_t>)               \
    _REGISTER_CUDA_REDOP(OFFSET + OP<int32_t>::REDOP_ID, OP<int32_t>)               \
    _REGISTER_CUDA_REDOP(OFFSET + OP<int64_t>::REDOP_ID, OP<int64_t>)               \
    _REGISTER_CUDA_REDOP(OFFSET + OP<uint8_t>::REDOP_ID, OP<uint8_t>)               \
    _REGISTER_CUDA_REDOP(OFFSET + OP<uint16_t>::REDOP_ID, OP<uint16_t>)             \
    _REGISTER_CUDA_REDOP(OFFSET + OP<uint32_t>::REDOP_ID, OP<uint32_t>)             \
    _REGISTER_CUDA_REDOP(OFFSET + OP<uint64_t>::REDOP_ID, OP<uint64_t>)             \
    _REGISTER_CUDA_REDOP(OFFSET + OP<bool>::REDOP_ID, OP<bool>)                     \
    _REGISTER_CUDA_REDOP(OFFSET + OP<__half>::REDOP_ID, OP<__half>)                 \
    _REGISTER_CUDA_REDOP(OFFSET + OP<complex<float>>::REDOP_ID, OP<complex<float>>) \
  }

void register_gpu_reduction_operators(ReductionOpID first_redop_id)
{
  REGISTER_CUDA_REDOPS(first_redop_id, ArgmaxReduction);
  REGISTER_CUDA_REDOPS(first_redop_id, ArgminReduction);
}

}  // namespace numpy
}  // namespace legate
