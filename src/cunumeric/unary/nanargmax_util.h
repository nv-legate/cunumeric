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

#pragma once

#include "cunumeric/cunumeric.h"
#include "cunumeric/arg.h"
#include "cunumeric/arg.inl"

namespace cunumeric {

enum class NanRedCode : int {
  // TODO: Right now, NanArgMax does just the ArgMax part...
  // so, we actually use Legate's MaxReduction
  NANARGMAX = CUNUMERIC_RED_NANARGMAX,
};

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) op_dispatch(NanRedCode op_code, Functor f, Fnargs&&... args)
{
  switch (op_code) {
    case NanRedCode::NANARGMAX:
      return f.template operator()<NanRedCode::NANARGMAX>(std::forward<Fnargs>(args)...);

    default: break;
  }

  assert(false);
  return f.template operator()<NanRedCode::NANARGMAX>(std::forward<Fnargs>(args)...);
}

template <NanRedCode OP_CODE, legate::LegateTypeCode TYPE_CODE>
struct NanRedOp {
  static constexpr bool valid = false;
};

template <legate::LegateTypeCode TYPE_CODE>
struct NanRedOp<NanRedCode::NANARGMAX, TYPE_CODE> {
  static constexpr bool valid = !legate::is_complex<TYPE_CODE>::value;

  using RHS = legate::legate_type_of<TYPE_CODE>;
  using VAL = RHS;
  using OP  = legate::MaxReduction<VAL>;
  // using OP  = ArgmaxReduction<VAL>;
};

}  // namespace cunumeric
