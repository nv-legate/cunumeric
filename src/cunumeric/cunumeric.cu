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

#include "cunumeric.h"
#include "arg.h"
#include "arg.inl"

namespace cunumeric {

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

#define _REGISTER_REDOP(ID, TYPE)                                                   \
  Runtime::register_reduction_op(                                                   \
    ID,                                                                             \
    Realm::ReductionOpUntyped::create_reduction_op<CUDAReductionOpWrapper<TYPE>>(), \
    NULL,                                                                           \
    NULL,                                                                           \
    false);

#define REGISTER_REDOPS(OP)                                                                        \
  {                                                                                                \
    _REGISTER_REDOP(context.get_reduction_op_id(OP<float>::REDOP_ID), OP<float>)                   \
    _REGISTER_REDOP(context.get_reduction_op_id(OP<double>::REDOP_ID), OP<double>)                 \
    _REGISTER_REDOP(context.get_reduction_op_id(OP<int8_t>::REDOP_ID), OP<int8_t>)                 \
    _REGISTER_REDOP(context.get_reduction_op_id(OP<int16_t>::REDOP_ID), OP<int16_t>)               \
    _REGISTER_REDOP(context.get_reduction_op_id(OP<int32_t>::REDOP_ID), OP<int32_t>)               \
    _REGISTER_REDOP(context.get_reduction_op_id(OP<int64_t>::REDOP_ID), OP<int64_t>)               \
    _REGISTER_REDOP(context.get_reduction_op_id(OP<uint8_t>::REDOP_ID), OP<uint8_t>)               \
    _REGISTER_REDOP(context.get_reduction_op_id(OP<uint16_t>::REDOP_ID), OP<uint16_t>)             \
    _REGISTER_REDOP(context.get_reduction_op_id(OP<uint32_t>::REDOP_ID), OP<uint32_t>)             \
    _REGISTER_REDOP(context.get_reduction_op_id(OP<uint64_t>::REDOP_ID), OP<uint64_t>)             \
    _REGISTER_REDOP(context.get_reduction_op_id(OP<bool>::REDOP_ID), OP<bool>)                     \
    _REGISTER_REDOP(context.get_reduction_op_id(OP<__half>::REDOP_ID), OP<__half>)                 \
    _REGISTER_REDOP(context.get_reduction_op_id(OP<complex<float>>::REDOP_ID), OP<complex<float>>) \
  }

void register_gpu_reduction_operators(legate::LibraryContext& context)
{
  REGISTER_REDOPS(ArgmaxReduction);
  REGISTER_REDOPS(ArgminReduction);
}

}  // namespace cunumeric
