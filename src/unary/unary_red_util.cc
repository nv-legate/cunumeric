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

#include "unary/unary_red_util.h"

namespace legate {
namespace numpy {

void deserialize(Deserializer& ctx, UnaryRedCode& code)
{
  int32_t value;
  deserialize(ctx, value);
  code = static_cast<UnaryRedCode>(value);
}

#define DECLARE_IDENTITY(OP_CODE, TYPE)                                               \
  template <>                                                                         \
  const typename UnaryRedOp<OP_CODE, TYPE>::VAL UnaryRedOp<OP_CODE, TYPE>::identity = \
    UnaryRedOp<OP_CODE, TYPE>::OP::identity;

#define DECLARE_IDENTITIES(OP_CODE)                    \
  DECLARE_IDENTITY(OP_CODE, LegateTypeCode::BOOL_LT)   \
  DECLARE_IDENTITY(OP_CODE, LegateTypeCode::INT8_LT)   \
  DECLARE_IDENTITY(OP_CODE, LegateTypeCode::INT16_LT)  \
  DECLARE_IDENTITY(OP_CODE, LegateTypeCode::INT32_LT)  \
  DECLARE_IDENTITY(OP_CODE, LegateTypeCode::INT64_LT)  \
  DECLARE_IDENTITY(OP_CODE, LegateTypeCode::UINT8_LT)  \
  DECLARE_IDENTITY(OP_CODE, LegateTypeCode::UINT16_LT) \
  DECLARE_IDENTITY(OP_CODE, LegateTypeCode::UINT32_LT) \
  DECLARE_IDENTITY(OP_CODE, LegateTypeCode::UINT64_LT) \
  DECLARE_IDENTITY(OP_CODE, LegateTypeCode::HALF_LT)   \
  DECLARE_IDENTITY(OP_CODE, LegateTypeCode::FLOAT_LT)  \
  DECLARE_IDENTITY(OP_CODE, LegateTypeCode::DOUBLE_LT) \
  DECLARE_IDENTITY(OP_CODE, LegateTypeCode::COMPLEX64_LT)
// TODO: We need to support reduction operators for complex<double>
// DECLARE_IDENTITY(LegateTypeCode::COMPLEX128_LT

DECLARE_IDENTITIES(UnaryRedCode::MAX)
DECLARE_IDENTITIES(UnaryRedCode::MIN)
DECLARE_IDENTITIES(UnaryRedCode::PROD)
DECLARE_IDENTITIES(UnaryRedCode::SUM)
// Sum reduction is available for complex<double>
DECLARE_IDENTITY(UnaryRedCode::SUM, LegateTypeCode::COMPLEX128_LT)

#undef DECLARE_IDENTITY
#undef DECLARE_IDENTITIES

template <>
const UntypedScalar UntypedScalarRedOp<UnaryRedCode::MAX>::identity = UntypedScalar();
template <>
const UntypedScalar UntypedScalarRedOp<UnaryRedCode::MIN>::identity = UntypedScalar();
template <>
const UntypedScalar UntypedScalarRedOp<UnaryRedCode::PROD>::identity = UntypedScalar();
template <>
const UntypedScalar UntypedScalarRedOp<UnaryRedCode::SUM>::identity = UntypedScalar();

}  // namespace numpy
}  // namespace legate
