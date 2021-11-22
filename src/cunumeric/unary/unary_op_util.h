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

#pragma once

#include "cunumeric/cunumeric.h"
#include "cunumeric/arg.h"

#include <math.h>

namespace cunumeric {

enum class UnaryOpCode : int {
  ABSOLUTE = 1,
  ARCCOS,
  ARCSIN,
  ARCTAN,
  CEIL,
  CLIP,
  COPY,
  COS,
  EXP,
  FLOOR,
  INVERT,
  ISINF,
  ISNAN,
  LOG,
  LOGICAL_NOT,
  NEGATIVE,
  SIN,
  SQRT,
  TAN,
  TANH,
  CONJ,
  REAL,
  IMAG,
  GETARG,
};

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) op_dispatch(UnaryOpCode op_code, Functor f, Fnargs&&... args)
{
  switch (op_code) {
    case UnaryOpCode::ABSOLUTE:
      return f.template operator()<UnaryOpCode::ABSOLUTE>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::ARCCOS:
      return f.template operator()<UnaryOpCode::ARCCOS>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::ARCSIN:
      return f.template operator()<UnaryOpCode::ARCSIN>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::ARCTAN:
      return f.template operator()<UnaryOpCode::ARCTAN>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::CEIL:
      return f.template operator()<UnaryOpCode::CEIL>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::CLIP:
      return f.template operator()<UnaryOpCode::CLIP>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::COPY:
      return f.template operator()<UnaryOpCode::COPY>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::COS:
      return f.template operator()<UnaryOpCode::COS>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::EXP:
      return f.template operator()<UnaryOpCode::EXP>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::FLOOR:
      return f.template operator()<UnaryOpCode::FLOOR>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::INVERT:
      return f.template operator()<UnaryOpCode::INVERT>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::ISINF:
      return f.template operator()<UnaryOpCode::ISINF>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::ISNAN:
      return f.template operator()<UnaryOpCode::ISNAN>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::LOG:
      return f.template operator()<UnaryOpCode::LOG>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::LOGICAL_NOT:
      return f.template operator()<UnaryOpCode::LOGICAL_NOT>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::NEGATIVE:
      return f.template operator()<UnaryOpCode::NEGATIVE>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::SIN:
      return f.template operator()<UnaryOpCode::SIN>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::SQRT:
      return f.template operator()<UnaryOpCode::SQRT>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::TAN:
      return f.template operator()<UnaryOpCode::TAN>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::TANH:
      return f.template operator()<UnaryOpCode::TANH>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::CONJ:
      return f.template operator()<UnaryOpCode::CONJ>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::REAL:
      return f.template operator()<UnaryOpCode::REAL>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::IMAG:
      return f.template operator()<UnaryOpCode::IMAG>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::GETARG:
      return f.template operator()<UnaryOpCode::GETARG>(std::forward<Fnargs>(args)...);
  }
  assert(false);
  return f.template operator()<UnaryOpCode::ABSOLUTE>(std::forward<Fnargs>(args)...);
}

template <UnaryOpCode OP_CODE, legate::LegateTypeCode CODE>
struct UnaryOp {
  static constexpr bool valid = false;
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::ABSOLUTE, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  template <
    typename _T                                                                    = T,
    std::enable_if_t<legate::is_complex<_T>::value or
                     (std::is_integral<_T>::value and std::is_signed<_T>::value)>* = nullptr>
  constexpr decltype(auto) operator()(const _T& x) const
  {
    return abs(x);
  }

  template <
    typename _T                                                                    = T,
    std::enable_if_t<std::is_integral<_T>::value and std::is_unsigned<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& x) const
  {
    return x;
  }

  template <
    typename _T                                                                        = T,
    std::enable_if_t<!legate::is_complex<_T>::value and !std::is_integral<_T>::value>* = nullptr>
  constexpr decltype(auto) operator()(const _T& x) const
  {
    using std::fabs;
    return fabs(x);
  }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::ARCCOS, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::acos;
    return acos(x);
  }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::ARCSIN, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::asin;
    return asin(x);
  }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::ARCTAN, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::atan;
    return atan(x);
  }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::CEIL, CODE> {
  static constexpr bool valid = legate::is_floating_point<CODE>::value;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::ceil;
    return ceil(x);
  }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::CLIP, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args)
  {
    assert(args.size() == 2);
    min = args[0].scalar<T>();
    max = args[1].scalar<T>();
  }

  constexpr T operator()(const T& x) const { return (x < min) ? min : (x > max) ? max : x; }

  T min;
  T max;
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::COPY, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr T operator()(const T& x) const { return x; }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::COS, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::cos;
    return cos(x);
  }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::EXP, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::exp;
    return exp(x);
  }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::FLOOR, CODE> {
  static constexpr bool valid = legate::is_floating_point<CODE>::value;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::floor;
    return floor(x);
  }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::INVERT, CODE> {
  static constexpr bool valid = legate::is_integral<CODE>::value;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr T operator()(const T& x) const { return ~x; }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::ISINF, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  template <typename _T = T, std::enable_if_t<!std::is_floating_point<_T>::value>* = nullptr>
  constexpr bool operator()(const T& x) const
  {
    return false;
  }

  template <typename _T = T, std::enable_if_t<std::is_floating_point<_T>::value>* = nullptr>
  constexpr bool operator()(const T& x) const
  {
    return std::isinf(x);
  }

  template <typename _T>
  constexpr bool operator()(const complex<_T>& x) const
  {
    return std::isinf(x.imag()) || std::isinf(x.real());
  }
};

template <>
struct UnaryOp<UnaryOpCode::ISINF, legate::LegateTypeCode::HALF_LT> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Store>& args) {}

  __CUDA_HD__ bool operator()(const __half& x) const { return isinf(x); }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::ISNAN, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  template <typename _T = T, std::enable_if_t<!std::is_floating_point<_T>::value>* = nullptr>
  constexpr bool operator()(const T& x) const
  {
    return false;
  }

  template <typename _T = T, std::enable_if_t<std::is_floating_point<_T>::value>* = nullptr>
  constexpr bool operator()(const T& x) const
  {
    using std::isnan;
    return isnan(x);
  }

  template <typename _T>
  constexpr bool operator()(const complex<_T>& x) const
  {
    return std::isnan(x.imag()) || std::isnan(x.real());
  }
};

template <>
struct UnaryOp<UnaryOpCode::ISNAN, legate::LegateTypeCode::HALF_LT> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Store>& args) {}

  __CUDA_HD__ bool operator()(const __half& x) const { return isnan(x); }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::LOG, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::log;
    return log(x);
  }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::LOGICAL_NOT, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  template <typename _T = T, std::enable_if_t<!legate::is_complex<_T>::value>* = nullptr>
  constexpr bool operator()(const T& x) const
  {
    return !static_cast<bool>(x);
  }

  template <typename _T = T, std::enable_if_t<legate::is_complex<_T>::value>* = nullptr>
  constexpr bool operator()(const T& x) const
  {
    return !static_cast<bool>(x.real());
  }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::NEGATIVE, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& x) const { return -x; }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::SIN, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::sin;
    return sin(x);
  }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::SQRT, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::sqrt;
    return sqrt(x);
  }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::TAN, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::tan;
    return tan(x);
  }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::TANH, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::legate_type_of<CODE>;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::tanh;
    return tanh(x);
  }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::CONJ, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = legate::is_complex<T>::value;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& x) const { return T{x.real(), -x.imag()}; }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::REAL, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = legate::is_complex<T>::value;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& x) const { return x.real(); }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::IMAG, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = legate::is_complex<T>::value;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& x) const { return x.imag(); }
};

template <legate::LegateTypeCode CODE>
struct UnaryOp<UnaryOpCode::GETARG, CODE> {
  using T                     = Argval<legate::legate_type_of<CODE>>;
  static constexpr bool valid = true;

  UnaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& x) const { return x.arg; }
};

}  // namespace cunumeric
