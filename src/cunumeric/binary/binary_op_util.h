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

namespace cunumeric {

enum class BinaryOpCode : int {
  ADD           = CUNUMERIC_BINOP_ADD,
  ALLCLOSE      = CUNUMERIC_BINOP_ALLCLOSE,
  ARCTAN2       = CUNUMERIC_BINOP_ARCTAN2,
  BITWISE_AND   = CUNUMERIC_BINOP_BITWISE_AND,
  BITWISE_OR    = CUNUMERIC_BINOP_BITWISE_OR,
  BITWISE_XOR   = CUNUMERIC_BINOP_BITWISE_XOR,
  COPYSIGN      = CUNUMERIC_BINOP_COPYSIGN,
  DIVIDE        = CUNUMERIC_BINOP_DIVIDE,
  EQUAL         = CUNUMERIC_BINOP_EQUAL,
  FLOAT_POWER   = CUNUMERIC_BINOP_FLOAT_POWER,
  FLOOR_DIVIDE  = CUNUMERIC_BINOP_FLOOR_DIVIDE,
  FMOD          = CUNUMERIC_BINOP_FMOD,
  GCD           = CUNUMERIC_BINOP_GCD,
  GREATER       = CUNUMERIC_BINOP_GREATER,
  GREATER_EQUAL = CUNUMERIC_BINOP_GREATER_EQUAL,
  HYPOT         = CUNUMERIC_BINOP_HYPOT,
  LCM           = CUNUMERIC_BINOP_LCM,
  LDEXP         = CUNUMERIC_BINOP_LDEXP,
  LEFT_SHIFT    = CUNUMERIC_BINOP_LEFT_SHIFT,
  LESS          = CUNUMERIC_BINOP_LESS,
  LESS_EQUAL    = CUNUMERIC_BINOP_LESS_EQUAL,
  LOGADDEXP     = CUNUMERIC_BINOP_LOGADDEXP,
  LOGADDEXP2    = CUNUMERIC_BINOP_LOGADDEXP2,
  LOGICAL_AND   = CUNUMERIC_BINOP_LOGICAL_AND,
  LOGICAL_OR    = CUNUMERIC_BINOP_LOGICAL_OR,
  LOGICAL_XOR   = CUNUMERIC_BINOP_LOGICAL_XOR,
  MAXIMUM       = CUNUMERIC_BINOP_MAXIMUM,
  MINIMUM       = CUNUMERIC_BINOP_MINIMUM,
  MOD           = CUNUMERIC_BINOP_MOD,
  MULTIPLY      = CUNUMERIC_BINOP_MULTIPLY,
  NEXTAFTER     = CUNUMERIC_BINOP_NEXTAFTER,
  NOT_EQUAL     = CUNUMERIC_BINOP_NOT_EQUAL,
  POWER         = CUNUMERIC_BINOP_POWER,
  RIGHT_SHIFT   = CUNUMERIC_BINOP_RIGHT_SHIFT,
  SUBTRACT      = CUNUMERIC_BINOP_SUBTRACT,
};

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) op_dispatch(BinaryOpCode op_code, Functor f, Fnargs&&... args)
{
  switch (op_code) {
    case BinaryOpCode::ADD:
      return f.template operator()<BinaryOpCode::ADD>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::ARCTAN2:
      return f.template operator()<BinaryOpCode::ARCTAN2>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::BITWISE_AND:
      return f.template operator()<BinaryOpCode::BITWISE_AND>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::BITWISE_OR:
      return f.template operator()<BinaryOpCode::BITWISE_OR>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::BITWISE_XOR:
      return f.template operator()<BinaryOpCode::BITWISE_XOR>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::COPYSIGN:
      return f.template operator()<BinaryOpCode::COPYSIGN>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::DIVIDE:
      return f.template operator()<BinaryOpCode::DIVIDE>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::EQUAL:
      return f.template operator()<BinaryOpCode::EQUAL>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::FLOAT_POWER:
      return f.template operator()<BinaryOpCode::FLOAT_POWER>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::FMOD:
      return f.template operator()<BinaryOpCode::FMOD>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::GCD:
      return f.template operator()<BinaryOpCode::GCD>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::FLOOR_DIVIDE:
      return f.template operator()<BinaryOpCode::FLOOR_DIVIDE>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::GREATER:
      return f.template operator()<BinaryOpCode::GREATER>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::GREATER_EQUAL:
      return f.template operator()<BinaryOpCode::GREATER_EQUAL>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::HYPOT:
      return f.template operator()<BinaryOpCode::HYPOT>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::LCM:
      return f.template operator()<BinaryOpCode::LCM>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::LDEXP:
      return f.template operator()<BinaryOpCode::LDEXP>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::LEFT_SHIFT:
      return f.template operator()<BinaryOpCode::LEFT_SHIFT>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::LESS:
      return f.template operator()<BinaryOpCode::LESS>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::LESS_EQUAL:
      return f.template operator()<BinaryOpCode::LESS_EQUAL>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::LOGADDEXP:
      return f.template operator()<BinaryOpCode::LOGADDEXP>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::LOGADDEXP2:
      return f.template operator()<BinaryOpCode::LOGADDEXP2>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::LOGICAL_AND:
      return f.template operator()<BinaryOpCode::LOGICAL_AND>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::LOGICAL_OR:
      return f.template operator()<BinaryOpCode::LOGICAL_OR>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::LOGICAL_XOR:
      return f.template operator()<BinaryOpCode::LOGICAL_XOR>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::MAXIMUM:
      return f.template operator()<BinaryOpCode::MAXIMUM>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::MINIMUM:
      return f.template operator()<BinaryOpCode::MINIMUM>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::MOD:
      return f.template operator()<BinaryOpCode::MOD>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::MULTIPLY:
      return f.template operator()<BinaryOpCode::MULTIPLY>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::NEXTAFTER:
      return f.template operator()<BinaryOpCode::NEXTAFTER>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::NOT_EQUAL:
      return f.template operator()<BinaryOpCode::NOT_EQUAL>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::POWER:
      return f.template operator()<BinaryOpCode::POWER>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::RIGHT_SHIFT:
      return f.template operator()<BinaryOpCode::RIGHT_SHIFT>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::SUBTRACT:
      return f.template operator()<BinaryOpCode::SUBTRACT>(std::forward<Fnargs>(args)...);
    default: break;
  }
  assert(false);
  return f.template operator()<BinaryOpCode::ADD>(std::forward<Fnargs>(args)...);
}

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) reduce_op_dispatch(BinaryOpCode op_code, Functor f, Fnargs&&... args)
{
  switch (op_code) {
    case BinaryOpCode::EQUAL:
      return f.template operator()<BinaryOpCode::EQUAL>(std::forward<Fnargs>(args)...);
    case BinaryOpCode::ALLCLOSE:
      return f.template operator()<BinaryOpCode::ALLCLOSE>(std::forward<Fnargs>(args)...);
    default: break;
  }
  assert(false);
  return f.template operator()<BinaryOpCode::EQUAL>(std::forward<Fnargs>(args)...);
}

template <BinaryOpCode OP_CODE, legate::LegateTypeCode CODE>
struct BinaryOp {
  static constexpr bool valid = false;
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::ADD, CODE> : std::plus<legate::legate_type_of<CODE>> {
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::ALLCLOSE, CODE> {
  using VAL                   = legate::legate_type_of<CODE>;
  static constexpr bool valid = true;

  BinaryOp(const std::vector<legate::Store>& args)
  {
    assert(args.size() == 2);
    rtol_ = args[0].scalar<double>();
    atol_ = args[1].scalar<double>();
  }

  template <typename T = VAL, std::enable_if_t<!legate::is_complex<T>::value>* = nullptr>
  constexpr bool operator()(const T& a, const T& b) const
  {
    using std::fabs;
    return fabs(static_cast<double>(a) - static_cast<double>(b)) <=
           atol_ + rtol_ * static_cast<double>(fabs(b));
  }

  template <typename T = VAL, std::enable_if_t<legate::is_complex<T>::value>* = nullptr>
  constexpr bool operator()(const T& a, const T& b) const
  {
    return static_cast<double>(abs(a - b)) <= atol_ + rtol_ * static_cast<double>(abs(b));
  }

  double rtol_{0};
  double atol_{0};
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::ARCTAN2, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = legate::is_floating_point<CODE>::value;

  BinaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& a, const T& b) const
  {
    using std::atan2;
    return atan2(a, b);
  }
};

template <>
struct BinaryOp<BinaryOpCode::ARCTAN2, legate::LegateTypeCode::HALF_LT> {
  using T                     = __half;
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}

  __CUDA_HD__ __half operator()(const __half& a, const __half& b) const
  {
    using std::atan2;
    return __half{atan2(static_cast<float>(a), static_cast<float>(b))};
  }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::BITWISE_AND, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = std::is_integral<T>::value;

  BinaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& a, const T& b) const
  {
    return std::bit_and<T>{}(a, b);
  }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::BITWISE_OR, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = std::is_integral<T>::value;

  BinaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& a, const T& b) const
  {
    return std::bit_or<T>{}(a, b);
  }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::BITWISE_XOR, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = std::is_integral<T>::value;

  BinaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& a, const T& b) const
  {
    return std::bit_xor<T>{}(a, b);
  }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::COPYSIGN, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = legate::is_floating_point<CODE>::value;
  BinaryOp(const std::vector<legate::Store>& args) {}

  constexpr T operator()(const T& a, const T& b) const
  {
    using std::copysign;
    return copysign(a, b);
  }
};

template <>
struct BinaryOp<BinaryOpCode::COPYSIGN, legate::LegateTypeCode::HALF_LT> {
  using T                     = __half;
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}

  __CUDA_HD__ __half operator()(const __half& a, const __half& b) const
  {
    using std::copysign;
    return __half{copysign(static_cast<float>(a), static_cast<float>(b))};
  }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::DIVIDE, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}

  template <typename _T = T, std::enable_if_t<std::is_integral<_T>::value>* = nullptr>
  constexpr double operator()(const T& a, const T& b) const
  {
    return static_cast<double>(a) / b;
  }

  template <typename _T = T, std::enable_if_t<!std::is_integral<_T>::value>* = nullptr>
  constexpr T operator()(const T& a, const T& b) const
  {
    return a / b;
  }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::EQUAL, CODE> : std::equal_to<legate::legate_type_of<CODE>> {
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::FLOAT_POWER, CODE> {
  using T = legate::legate_type_of<CODE>;
  static constexpr bool valid =
    CODE == legate::LegateTypeCode::DOUBLE_LT or CODE == legate::LegateTypeCode::COMPLEX128_LT;
  BinaryOp(const std::vector<legate::Store>& args) {}

  constexpr T operator()(const T& a, const T& b) const
  {
    using std::pow;
    return pow(a, b);
  }
};

template <>
struct BinaryOp<BinaryOpCode::FLOAT_POWER, legate::LegateTypeCode::COMPLEX64_LT> {
  using T                     = complex<float>;
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}

  __CUDA_HD__ complex<double> operator()(const complex<float>& a, const complex<float>& b) const
  {
    using std::pow;
    return pow(static_cast<complex<double>>(a), static_cast<complex<double>>(b));
  }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::FMOD, CODE> {
  using T = legate::legate_type_of<CODE>;
  static constexpr bool valid =
    not(CODE == legate::LegateTypeCode::BOOL_LT or legate::is_complex<T>::value);
  BinaryOp(const std::vector<legate::Store>& args) {}

  template <typename _T = T, std::enable_if_t<std::is_integral<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& a, const _T& b) const
  {
    return a % b;
  }

  template <typename _T = T, std::enable_if_t<!std::is_integral<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& a, const _T& b) const
  {
    using std::fmod;
    return fmod(a, b);
  }
};

template <>
struct BinaryOp<BinaryOpCode::FMOD, legate::LegateTypeCode::HALF_LT> {
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}
  __CUDA_HD__ __half operator()(const __half& a, const __half& b) const
  {
    using std::fmod;
    return __half{fmod(static_cast<float>(a), static_cast<float>(b))};
  }
};

template <typename T, std::enable_if_t<std::is_signed<T>::value>* = nullptr>
static __CUDA_HD__ T _gcd(T a, T b)
{
  T r;
  while (b != 0) {
    r = a % b;
    a = b;
    b = r;
  }
  return a >= 0 ? a : -a;
}

template <typename T, std::enable_if_t<!std::is_signed<T>::value>* = nullptr>
static __CUDA_HD__ T _gcd(T a, T b)
{
  T r;
  while (b != 0) {
    r = a % b;
    a = b;
    b = r;
  }
  return a;
}

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::GCD, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = std::is_integral<T>::value;
  BinaryOp(const std::vector<legate::Store>& args) {}

  __CUDA_HD__ T operator()(const T& a, const T& b) const { return _gcd(a, b); }
};

template <typename T,
          std::enable_if_t<std::is_integral<T>::value and std::is_signed<T>::value>* = nullptr>
static constexpr T floor_divide_signed(const T& a, const T& b)
{
  auto q = a / b;
  return q - (((a < 0) != (b < 0)) && q * b != a);
}

using std::floor;
template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::FLOOR_DIVIDE, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}

  template <typename _T                                                                  = T,
            std::enable_if_t<std::is_integral<_T>::value and std::is_signed<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& a, const _T& b) const
  {
    return floor_divide_signed(a, b);
  }

  template <typename _T                                                                   = T,
            std::enable_if_t<std::is_integral<_T>::value and !std::is_signed<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& a, const _T& b) const
  {
    return a / b;
  }

  template <typename _T = T, std::enable_if_t<!std::is_integral<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& a, const _T& b) const
  {
    return floor(a / b);
  }
};

template <>
struct BinaryOp<BinaryOpCode::FLOOR_DIVIDE, legate::LegateTypeCode::COMPLEX64_LT> {
  static constexpr bool valid = false;
  BinaryOp(const std::vector<legate::Store>& args) {}
};

template <>
struct BinaryOp<BinaryOpCode::FLOOR_DIVIDE, legate::LegateTypeCode::COMPLEX128_LT> {
  static constexpr bool valid = false;
  BinaryOp(const std::vector<legate::Store>& args) {}
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::GREATER, CODE> : std::greater<legate::legate_type_of<CODE>> {
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::GREATER_EQUAL, CODE>
  : std::greater_equal<legate::legate_type_of<CODE>> {
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::HYPOT, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = legate::is_floating_point<CODE>::value;

  BinaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& a, const T& b) const
  {
    using std::hypot;
    return hypot(a, b);
  }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::LCM, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = std::is_integral<T>::value;
  BinaryOp(const std::vector<legate::Store>& args) {}

  template <typename _T = T, std::enable_if_t<std::is_signed<_T>::value>* = nullptr>
  __CUDA_HD__ T operator()(const T& a, const T& b) const
  {
    T r = _gcd(a, b);
    if (r == 0) return 0;
    r = a / r * b;
    return r >= 0 ? r : -r;
  }

  template <typename _T = T, std::enable_if_t<!std::is_signed<_T>::value>* = nullptr>
  __CUDA_HD__ T operator()(const T& a, const T& b) const
  {
    T r = _gcd(a, b);
    if (r == 0) return 0;
    r = a / r * b;
    return r;
  }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::LDEXP, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = legate::is_floating_point<CODE>::value;
  BinaryOp(const std::vector<legate::Store>& args) {}

  __CUDA_HD__ T operator()(const T& a, const int32_t& b) const
  {
    using std::ldexp;
    return ldexp(a, b);
  }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::LEFT_SHIFT, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = CODE != BOOL_LT && std::is_integral<T>::value;

  BinaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& a, const T& b) const
  {
#if defined(__NVCC__) || defined(__CUDACC__)
    return a << b;
#else
    return (a << b) * (b >= 0);
#endif
  }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::LESS, CODE> : std::less<legate::legate_type_of<CODE>> {
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::LESS_EQUAL, CODE> : std::less_equal<legate::legate_type_of<CODE>> {
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::LOGADDEXP, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = legate::is_floating_point<CODE>::value;

  BinaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& a, const T& b) const
  {
    using std::exp;
    using std::fabs;
    using std::fmax;
    using std::log;
    using std::log1p;
    if (a == b)
      return a + log(T{2.0});
    else
      return fmax(a, b) + log1p(exp(-fabs(a - b)));
  }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::LOGADDEXP2, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = legate::is_floating_point<CODE>::value;

  BinaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& a, const T& b) const
  {
    using std::exp2;
    using std::fabs;
    using std::fmax;
    using std::log2;
    if (a == b)
      return a + T{1.0};
    else
      return fmax(a, b) + log2(T{1} + exp2(-fabs(a - b)));
  }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::LOGICAL_AND, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}

  template <typename _T = T, std::enable_if_t<legate::is_complex<_T>::value>* = nullptr>
  constexpr bool operator()(const _T& a, const _T& b) const
  {
    return static_cast<bool>(a.real()) && static_cast<bool>(b.real());
  }

  template <typename _T = T, std::enable_if_t<!legate::is_complex<_T>::value>* = nullptr>
  constexpr bool operator()(const _T& a, const _T& b) const
  {
    return static_cast<bool>(a) && static_cast<bool>(b);
  }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::LOGICAL_OR, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = true;

  BinaryOp(const std::vector<legate::Store>& args) {}

  template <typename _T = T, std::enable_if_t<legate::is_complex<_T>::value>* = nullptr>
  constexpr bool operator()(const _T& a, const _T& b) const
  {
    return static_cast<bool>(a.real()) || static_cast<bool>(b.real());
  }

  template <typename _T = T, std::enable_if_t<!legate::is_complex<_T>::value>* = nullptr>
  constexpr bool operator()(const _T& a, const _T& b) const
  {
    return static_cast<bool>(a) || static_cast<bool>(b);
  }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::LOGICAL_XOR, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}

  template <typename _T = T, std::enable_if_t<legate::is_complex<_T>::value>* = nullptr>
  constexpr bool operator()(const _T& a, const _T& b) const
  {
    return static_cast<bool>(a.real()) != static_cast<bool>(b.real());
  }

  template <typename _T = T, std::enable_if_t<!legate::is_complex<_T>::value>* = nullptr>
  constexpr bool operator()(const _T& a, const _T& b) const
  {
    return static_cast<bool>(a) != static_cast<bool>(b);
  }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::MAXIMUM, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}
  constexpr T operator()(const T& a, const T& b) const { return std::max<T>(a, b); }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::MINIMUM, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}
  constexpr T operator()(const T& a, const T& b) const { return std::min<T>(a, b); }
};

template <typename T>
constexpr T real_mod(const T& a, const T& b)
{
  T res = std::fmod(a, b);
  if (res) {
    if ((b < static_cast<T>(0)) != (res < static_cast<T>(0))) res += b;
  } else {
    res = std::copysign(static_cast<T>(0), b);
  }
  return res;
}

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::MOD, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}

  template <typename _T                                                                  = T,
            std::enable_if_t<std::is_integral<_T>::value and std::is_signed<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& a, const _T& b) const
  {
    auto q = floor_divide_signed(a, b);
    return (a - b * q) * (b != 0);
  }

  template <typename _T                                                                   = T,
            std::enable_if_t<std::is_integral<_T>::value and !std::is_signed<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& a, const _T& b) const
  {
    return a % b;
  }

  template <typename _T = T, std::enable_if_t<!std::is_integral<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& a, const _T& b) const
  {
    return real_mod(a, b);
  }
};

template <>
struct BinaryOp<BinaryOpCode::MOD, legate::LegateTypeCode::HALF_LT> {
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}
  __CUDA_HD__ __half operator()(const __half& a, const __half& b) const
  {
    return static_cast<__half>(real_mod(static_cast<float>(a), static_cast<float>(b)));
  }
};

template <>
struct BinaryOp<BinaryOpCode::MOD, legate::LegateTypeCode::COMPLEX64_LT> {
  static constexpr bool valid = false;
  BinaryOp(const std::vector<legate::Store>& args) {}
};

template <>
struct BinaryOp<BinaryOpCode::MOD, legate::LegateTypeCode::COMPLEX128_LT> {
  static constexpr bool valid = false;
  BinaryOp(const std::vector<legate::Store>& args) {}
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::MULTIPLY, CODE> : std::multiplies<legate::legate_type_of<CODE>> {
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::NEXTAFTER, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = legate::is_floating_point<CODE>::value;
  BinaryOp(const std::vector<legate::Store>& args) {}

  constexpr T operator()(const T& a, const T& b) const
  {
    using std::nextafter;
    return nextafter(a, b);
  }
};

template <>
struct BinaryOp<BinaryOpCode::NEXTAFTER, legate::LegateTypeCode::HALF_LT> {
  using T                     = __half;
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}

  __CUDA_HD__ __half operator()(const __half& a, const __half& b) const
  {
    using std::nextafter;
    return __half{nextafter(static_cast<float>(a), static_cast<float>(b))};
  }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::NOT_EQUAL, CODE> : std::not_equal_to<legate::legate_type_of<CODE>> {
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::POWER, CODE> {
  using VAL                   = legate::legate_type_of<CODE>;
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}
  constexpr VAL operator()(const VAL& a, const VAL& b) const
  {
    return std::pow(static_cast<double>(a), static_cast<double>(b));
  }
};

template <>
struct BinaryOp<BinaryOpCode::POWER, legate::LegateTypeCode::HALF_LT> {
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}
  __CUDA_HD__ __half operator()(const __half& a, const __half& b) const { return pow(a, b); }
};

template <>
struct BinaryOp<BinaryOpCode::POWER, legate::LegateTypeCode::COMPLEX64_LT> {
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}
  __CUDA_HD__ complex<float> operator()(const complex<float>& a, const complex<float>& b) const
  {
    return pow(a, b);
  }
};

template <>
struct BinaryOp<BinaryOpCode::POWER, legate::LegateTypeCode::COMPLEX128_LT> {
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}
  __CUDA_HD__ complex<double> operator()(const complex<double>& a, const complex<double>& b) const
  {
    return pow(a, b);
  }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::RIGHT_SHIFT, CODE> {
  using T                     = legate::legate_type_of<CODE>;
  static constexpr bool valid = CODE != BOOL_LT && std::is_integral<T>::value;

  BinaryOp(const std::vector<legate::Store>& args) {}

  constexpr decltype(auto) operator()(const T& a, const T& b) const { return a >> b; }
};

template <legate::LegateTypeCode CODE>
struct BinaryOp<BinaryOpCode::SUBTRACT, CODE> : std::minus<legate::legate_type_of<CODE>> {
  static constexpr bool valid = true;
  BinaryOp(const std::vector<legate::Store>& args) {}
};

template <BinaryOpCode OP_CODE, legate::LegateTypeCode CODE>
struct RHS2OfBinaryOp {
  using type = legate::legate_type_of<CODE>;
};

template <legate::LegateTypeCode CODE>
struct RHS2OfBinaryOp<BinaryOpCode::LDEXP, CODE> {
  using type = int32_t;
};

template <BinaryOpCode OP_CODE, legate::LegateTypeCode CODE>
using rhs2_of_binary_op = typename RHS2OfBinaryOp<OP_CODE, CODE>::type;

}  // namespace cunumeric
