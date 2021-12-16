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

#include "cunumeric/arg.h"

using namespace Legion;

namespace cunumeric {

#define DEFINE_ARGMAX_IDENTITY(TYPE) \
  template <>                        \
  const Argval<TYPE> ArgmaxReduction<TYPE>::identity = Argval<TYPE>(MaxReduction<TYPE>::identity);

#define DEFINE_ARGMIN_IDENTITY(TYPE) \
  template <>                        \
  const Argval<TYPE> ArgminReduction<TYPE>::identity = Argval<TYPE>(MinReduction<TYPE>::identity);

#define DEFINE_IDENTITIES(TYPE) \
  DEFINE_ARGMAX_IDENTITY(TYPE)  \
  DEFINE_ARGMIN_IDENTITY(TYPE)

DEFINE_IDENTITIES(__half)
DEFINE_IDENTITIES(float)
DEFINE_IDENTITIES(double)
DEFINE_IDENTITIES(bool)
DEFINE_IDENTITIES(int8_t)
DEFINE_IDENTITIES(int16_t)
DEFINE_IDENTITIES(int32_t)
DEFINE_IDENTITIES(int64_t)
DEFINE_IDENTITIES(uint8_t)
DEFINE_IDENTITIES(uint16_t)
DEFINE_IDENTITIES(uint32_t)
DEFINE_IDENTITIES(uint64_t)
DEFINE_IDENTITIES(complex<float>)

#define _REGISTER_REDOP(ID, TYPE) Runtime::register_reduction_op<TYPE>(ID);

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

void register_cpu_reduction_operators(legate::LibraryContext& context)
{
  REGISTER_REDOPS(ArgmaxReduction);
  REGISTER_REDOPS(ArgminReduction);
}

}  // namespace cunumeric
