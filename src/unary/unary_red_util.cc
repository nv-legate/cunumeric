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

template <>
const UntypedScalar UntypedScalarRedOp<UnaryRedCode::MAX>::identity = UntypedScalar();
template <>
const UntypedScalar UntypedScalarRedOp<UnaryRedCode::MIN>::identity = UntypedScalar();
template <>
const UntypedScalar UntypedScalarRedOp<UnaryRedCode::PROD>::identity = UntypedScalar();
template <>
const UntypedScalar UntypedScalarRedOp<UnaryRedCode::SUM>::identity = UntypedScalar();
template <>
const UntypedScalar UntypedScalarRedOp<UnaryRedCode::ARGMAX>::identity = UntypedScalar();
template <>
const UntypedScalar UntypedScalarRedOp<UnaryRedCode::ARGMIN>::identity = UntypedScalar();

}  // namespace numpy
}  // namespace legate
