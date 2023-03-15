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

enum class UnaryRedCode : int {
  ALL           = CUNUMERIC_RED_ALL,
  ANY           = CUNUMERIC_RED_ANY,
  ARGMAX        = CUNUMERIC_RED_ARGMAX,
  ARGMIN        = CUNUMERIC_RED_ARGMIN,
  CONTAINS      = CUNUMERIC_RED_CONTAINS,
  COUNT_NONZERO = CUNUMERIC_RED_COUNT_NONZERO,
  MAX           = CUNUMERIC_RED_MAX,
  MIN           = CUNUMERIC_RED_MIN,
  PROD          = CUNUMERIC_RED_PROD,
  SUM           = CUNUMERIC_RED_SUM,
};

template <UnaryRedCode OP_CODE>
struct is_arg_reduce : std::false_type {};
template <>
struct is_arg_reduce<UnaryRedCode::ARGMAX> : std::true_type {};
template <>
struct is_arg_reduce<UnaryRedCode::ARGMIN> : std::true_type {};

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) op_dispatch(UnaryRedCode op_code, Functor f, Fnargs&&... args)
{
  switch (op_code) {
    case UnaryRedCode::ALL:
      return f.template operator()<UnaryRedCode::ALL>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::ANY:
      return f.template operator()<UnaryRedCode::ANY>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::ARGMAX:
      return f.template operator()<UnaryRedCode::ARGMAX>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::ARGMIN:
      return f.template operator()<UnaryRedCode::ARGMIN>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::CONTAINS:
      return f.template operator()<UnaryRedCode::CONTAINS>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::COUNT_NONZERO:
      return f.template operator()<UnaryRedCode::COUNT_NONZERO>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::MAX:
      return f.template operator()<UnaryRedCode::MAX>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::MIN:
      return f.template operator()<UnaryRedCode::MIN>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::PROD:
      return f.template operator()<UnaryRedCode::PROD>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::SUM:
      return f.template operator()<UnaryRedCode::SUM>(std::forward<Fnargs>(args)...);
    default: break;
  }
  assert(false);
  return f.template operator()<UnaryRedCode::MAX>(std::forward<Fnargs>(args)...);
}

template <UnaryRedCode OP_CODE, legate::LegateTypeCode TYPE_CODE>
struct UnaryRedOp {
  static constexpr bool valid = false;
};

template <legate::LegateTypeCode TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::ALL, TYPE_CODE> {
  static constexpr bool valid = TYPE_CODE != legate::LegateTypeCode::COMPLEX128_LT;

  using RHS = legate::legate_type_of<TYPE_CODE>;
  using VAL = bool;
  using OP  = legate::ProdReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>&, int32_t, const RHS& rhs)
  {
    return rhs != RHS(0);
  }

  __CUDA_HD__ static VAL convert(const RHS& rhs) { return rhs != RHS(0); }
};

template <legate::LegateTypeCode TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::ANY, TYPE_CODE> {
  static constexpr bool valid = TYPE_CODE != legate::LegateTypeCode::COMPLEX128_LT;

  using RHS = legate::legate_type_of<TYPE_CODE>;
  using VAL = bool;
  using OP  = legate::SumReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>&, int32_t, const RHS& rhs)
  {
    return rhs != RHS(0);
  }

  __CUDA_HD__ static VAL convert(const RHS& rhs) { return rhs != RHS(0); }
};

template <legate::LegateTypeCode TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::COUNT_NONZERO, TYPE_CODE> {
  static constexpr bool valid = true;

  using RHS = legate::legate_type_of<TYPE_CODE>;
  using VAL = uint64_t;
  using OP  = legate::SumReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>&, int32_t, const RHS& rhs)
  {
    return static_cast<VAL>(rhs != RHS(0));
  }

  __CUDA_HD__ static VAL convert(const RHS& rhs) { return static_cast<VAL>(rhs != RHS(0)); }
};

template <legate::LegateTypeCode TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::MAX, TYPE_CODE> {
  static constexpr bool valid = !legate::is_complex<TYPE_CODE>::value;

  using RHS = legate::legate_type_of<TYPE_CODE>;
  using VAL = RHS;
  using OP  = legate::MaxReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>&, int32_t, const RHS& rhs)
  {
    return rhs;
  }

  __CUDA_HD__ static VAL convert(const RHS& rhs) { return rhs; }
};

template <legate::LegateTypeCode TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::MIN, TYPE_CODE> {
  static constexpr bool valid = !legate::is_complex<TYPE_CODE>::value;

  using RHS = legate::legate_type_of<TYPE_CODE>;
  using VAL = RHS;
  using OP  = legate::MinReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>&, int32_t, const RHS& rhs)
  {
    return rhs;
  }

  __CUDA_HD__ static VAL convert(const RHS& rhs) { return rhs; }
};

template <legate::LegateTypeCode TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::PROD, TYPE_CODE> {
  static constexpr bool valid = TYPE_CODE != legate::LegateTypeCode::COMPLEX128_LT;

  using RHS = legate::legate_type_of<TYPE_CODE>;
  using VAL = RHS;
  using OP  = legate::ProdReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>&, int32_t, const RHS& rhs)
  {
    return rhs;
  }

  __CUDA_HD__ static VAL convert(const RHS& rhs) { return rhs; }
};

template <legate::LegateTypeCode TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::SUM, TYPE_CODE> {
  static constexpr bool valid = true;

  using RHS = legate::legate_type_of<TYPE_CODE>;
  using VAL = RHS;
  using OP  = legate::SumReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>&, int32_t, const RHS& rhs)
  {
    return rhs;
  }

  __CUDA_HD__ static VAL convert(const RHS& rhs) { return rhs; }
};

template <legate::LegateTypeCode TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::ARGMAX, TYPE_CODE> {
  static constexpr bool valid = !legate::is_complex<TYPE_CODE>::value;

  using RHS = legate::legate_type_of<TYPE_CODE>;
  using VAL = Argval<RHS>;
  using OP  = ArgmaxReduction<RHS>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>& point,
                                 int32_t collapsed_dim,
                                 const RHS& rhs)
  {
    return VAL(point[collapsed_dim], rhs);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>& point,
                                 const legate::Point<DIM>& shape,
                                 const RHS& rhs)
  {
    int64_t idx = 0;
    for (int32_t dim = 0; dim < DIM; ++dim) idx = idx * shape[dim] + point[dim];
    return VAL(idx, rhs);
  }
};

template <legate::LegateTypeCode TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::ARGMIN, TYPE_CODE> {
  static constexpr bool valid = !legate::is_complex<TYPE_CODE>::value;

  using RHS = legate::legate_type_of<TYPE_CODE>;
  using VAL = Argval<RHS>;
  using OP  = ArgminReduction<RHS>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>& point,
                                 int32_t collapsed_dim,
                                 const RHS& rhs)
  {
    return VAL(point[collapsed_dim], rhs);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>& point,
                                 const legate::Point<DIM>& shape,
                                 const RHS& rhs)
  {
    int64_t idx = 0;
    for (int32_t dim = 0; dim < DIM; ++dim) idx = idx * shape[dim] + point[dim];
    return VAL(idx, rhs);
  }
};

template <legate::LegateTypeCode TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::CONTAINS, TYPE_CODE> {
  // Set to false so that this only gets enabled when expliclty declared valid.
  static constexpr bool valid = false;
  // This class only provides the typedefs necessary to match the other operators.
  // It does not provide fold/convert functions.
  using RHS     = legate::legate_type_of<TYPE_CODE>;
  using VAL     = bool;
  using _RED_OP = UnaryRedOp<UnaryRedCode::SUM, legate::LegateTypeCode::BOOL_LT>;
  using OP      = _RED_OP::OP;
};

}  // namespace cunumeric
