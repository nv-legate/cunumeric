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

#include "cunumeric/arg_redop_register.h"

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

#undef DEFINE_ARGMAX_IDENTITY
#undef DEFINE_ARGMIN_IDENTITY
#undef DEFINE_IDENTITIES

/*static*/ int32_t register_reduction_op_fn::register_reduction_op_fn::next_reduction_operator_id()
{
  static int32_t next_redop_id = 0;
  return next_redop_id++;
}

}  // namespace cunumeric

#ifndef LEGATE_USE_CUDA

extern "C" {

void cunumeric_register_reduction_op(int32_t type_uid, int32_t _elem_type_code)
{
  auto elem_type_code = static_cast<legate::Type::Code>(_elem_type_code);
  legate::type_dispatch(elem_type_code, cunumeric::register_reduction_op_fn{}, type_uid);
}
}

#endif
