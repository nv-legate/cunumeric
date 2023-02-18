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

#include "cunumeric/arg.h"
#include "cunumeric/arg.inl"

namespace cunumeric {

#define DEFINE_ARGMAX_IDENTITY(TYPE)                   \
  template <>                                          \
  const Argval<TYPE> ArgmaxReduction<TYPE>::identity = \
    Argval<TYPE>(legate::MaxReduction<TYPE>::identity);

#define DEFINE_ARGMIN_IDENTITY(TYPE)                   \
  template <>                                          \
  const Argval<TYPE> ArgminReduction<TYPE>::identity = \
    Argval<TYPE>(legate::MinReduction<TYPE>::identity);

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

#ifndef LEGATE_USE_CUDA

#define REGISTER_REDOPS(OP)                              \
  {                                                      \
    context.register_reduction_operator<OP<float>>();    \
    context.register_reduction_operator<OP<double>>();   \
    context.register_reduction_operator<OP<int8_t>>();   \
    context.register_reduction_operator<OP<int16_t>>();  \
    context.register_reduction_operator<OP<int32_t>>();  \
    context.register_reduction_operator<OP<int64_t>>();  \
    context.register_reduction_operator<OP<uint8_t>>();  \
    context.register_reduction_operator<OP<uint16_t>>(); \
    context.register_reduction_operator<OP<uint32_t>>(); \
    context.register_reduction_operator<OP<uint64_t>>(); \
    context.register_reduction_operator<OP<bool>>();     \
    context.register_reduction_operator<OP<__half>>();   \
  }

void register_reduction_operators(legate::LibraryContext& context)
{
  REGISTER_REDOPS(ArgmaxReduction);
  REGISTER_REDOPS(ArgminReduction);
}

#endif

}  // namespace cunumeric
