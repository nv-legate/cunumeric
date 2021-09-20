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

#include "numpy.h"

namespace legate {
namespace numpy {

enum class DoubleBinaryOpCode : int {
  ADD = 1,
  DIVIDE,
  EQUAL,
  FLOOR_DIVIDE,
  GREATER,
  GREATER_EQUAL,
  LESS,
  LESS_EQUAL,
  MAXIMUM,
  MINIMUM,
  MOD,
  MULTIPLY,
  NOT_EQUAL,
  POWER,
  SUBTRACT,
  ALLCLOSE,
};

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) op_dispatch(DoubleBinaryOpCode op_code, Functor f, Fnargs&&... args)
{
  switch (op_code) {
    case DoubleBinaryOpCode::ADD:
      return f.template operator()<DoubleBinaryOpCode::ADD>(std::forward<Fnargs>(args)...);
    case DoubleBinaryOpCode::DIVIDE:
      return f.template operator()<DoubleBinaryOpCode::DIVIDE>(std::forward<Fnargs>(args)...);
    //case DoubleBinaryOpCode::EQUAL:
    //  return f.template operator()<DoubleBinaryOpCode::EQUAL>(std::forward<Fnargs>(args)...);
    case DoubleBinaryOpCode::FLOOR_DIVIDE:
      return f.template operator()<DoubleBinaryOpCode::FLOOR_DIVIDE>(std::forward<Fnargs>(args)...);
    //case DoubleBinaryOpCode::GREATER:
    //  return f.template operator()<DoubleBinaryOpCode::GREATER>(std::forward<Fnargs>(args)...);
    //case DoubleBinaryOpCode::GREATER_EQUAL:
    //  return f.template operator()<DoubleBinaryOpCode::GREATER_EQUAL>(std::forward<Fnargs>(args)...);
    //case DoubleBinaryOpCode::LESS:
    //  return f.template operator()<DoubleBinaryOpCode::LESS>(std::forward<Fnargs>(args)...);
    //case DoubleBinaryOpCode::LESS_EQUAL:
    //  return f.template operator()<DoubleBinaryOpCode::LESS_EQUAL>(std::forward<Fnargs>(args)...);
    case DoubleBinaryOpCode::MAXIMUM:
      return f.template operator()<DoubleBinaryOpCode::MAXIMUM>(std::forward<Fnargs>(args)...);
    case DoubleBinaryOpCode::MINIMUM:
      return f.template operator()<DoubleBinaryOpCode::MINIMUM>(std::forward<Fnargs>(args)...);
    case DoubleBinaryOpCode::MOD:
      return f.template operator()<DoubleBinaryOpCode::MOD>(std::forward<Fnargs>(args)...);
    case DoubleBinaryOpCode::MULTIPLY:
      return f.template operator()<DoubleBinaryOpCode::MULTIPLY>(std::forward<Fnargs>(args)...);
    //case DoubleBinaryOpCode::NOT_EQUAL:
    //  return f.template operator()<DoubleBinaryOpCode::NOT_EQUAL>(std::forward<Fnargs>(args)...);
    case DoubleBinaryOpCode::POWER:
      return f.template operator()<DoubleBinaryOpCode::POWER>(std::forward<Fnargs>(args)...);
    case DoubleBinaryOpCode::SUBTRACT:
      return f.template operator()<DoubleBinaryOpCode::SUBTRACT>(std::forward<Fnargs>(args)...);
    default: break;
  }
  assert(false);
  return f.template operator()<DoubleBinaryOpCode::ADD>(std::forward<Fnargs>(args)...);
}

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) reduce_op_dispatch(DoubleBinaryOpCode op_code, Functor f, Fnargs&&... args)
{
  switch (op_code) {
    //case DoubleBinaryOpCode::EQUAL:
    //  return f.template operator()<DoubleBinaryOpCode::EQUAL>(std::forward<Fnargs>(args)...);
    //case DoubleBinaryOpCode::ALLCLOSE:
    //  return f.template operator()<DoubleBinaryOpCode::ALLCLOSE>(std::forward<Fnargs>(args)...);
    default: break;
  }
  assert(false);
  //return f.template operator()<DoubleBinaryOpCode::EQUAL>(std::forward<Fnargs>(args)...);
}

template <DoubleBinaryOpCode OP_CODE, LegateTypeCode CODE>
struct DoubleBinaryOp {
  static constexpr bool valid = false;
};

template <LegateTypeCode CODE>
struct DoubleBinaryOp<DoubleBinaryOpCode::ADD, CODE> : std::plus<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
};

template <LegateTypeCode CODE>
struct DoubleBinaryOp<DoubleBinaryOpCode::DIVIDE, CODE> : std::divides<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
};
/*
template <LegateTypeCode CODE>
struct DoubleBinaryOp<DoubleBinaryOpCode::EQUAL, CODE> : std::equal_to<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
};
*/
using std::floor;
template <LegateTypeCode CODE>
struct DoubleBinaryOp<DoubleBinaryOpCode::FLOOR_DIVIDE, CODE> {
  using T                     = legate_type_of<CODE>;
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
  constexpr T operator()(const T& a, const T& b) const { return floor(a / b); }
};

template <>
struct DoubleBinaryOp<DoubleBinaryOpCode::FLOOR_DIVIDE, LegateTypeCode::COMPLEX64_LT> {
  static constexpr bool valid = false;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
};

template <>
struct DoubleBinaryOp<DoubleBinaryOpCode::FLOOR_DIVIDE, LegateTypeCode::COMPLEX128_LT> {
  static constexpr bool valid = false;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
};
/*
template <LegateTypeCode CODE>
struct DoubleBinaryOp<DoubleBinaryOpCode::GREATER, CODE> : std::greater<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
};

template <LegateTypeCode CODE>
struct DoubleBinaryOp<DoubleBinaryOpCode::GREATER_EQUAL, CODE> : std::greater_equal<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
};

template <LegateTypeCode CODE>
struct DoubleBinaryOp<DoubleBinaryOpCode::LESS, CODE> : std::less<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
};

template <LegateTypeCode CODE>
struct DoubleBinaryOp<DoubleBinaryOpCode::LESS_EQUAL, CODE> : std::less_equal<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
};
*/
template <LegateTypeCode CODE>
struct DoubleBinaryOp<DoubleBinaryOpCode::MAXIMUM, CODE> {
  using T                     = legate_type_of<CODE>;
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
  constexpr T operator()(const T& a, const T& b) const { return std::max<T>(a, b); }
};

template <LegateTypeCode CODE>
struct DoubleBinaryOp<DoubleBinaryOpCode::MINIMUM, CODE> {
  using T                     = legate_type_of<CODE>;
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
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

template <LegateTypeCode CODE>
struct DoubleBinaryOp<DoubleBinaryOpCode::MOD, CODE> {
  using T                     = legate_type_of<CODE>;
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
  template <typename _T = T, std::enable_if_t<std::is_integral<_T>::value>* = nullptr>
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
struct DoubleBinaryOp<DoubleBinaryOpCode::MOD, LegateTypeCode::HALF_LT> {
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
  LEGATE_DEVICE_PREFIX __half operator()(const __half& a, const __half& b) const
  {
    return static_cast<__half>(real_mod(static_cast<float>(a), static_cast<float>(b)));
  }
};

template <>
struct DoubleBinaryOp<DoubleBinaryOpCode::MOD, LegateTypeCode::COMPLEX64_LT> {
  static constexpr bool valid = false;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
};

template <>
struct DoubleBinaryOp<DoubleBinaryOpCode::MOD, LegateTypeCode::COMPLEX128_LT> {
  static constexpr bool valid = false;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
};

template <LegateTypeCode CODE>
struct DoubleBinaryOp<DoubleBinaryOpCode::MULTIPLY, CODE> : std::multiplies<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
};
/*
template <LegateTypeCode CODE>
struct DoubleBinaryOp<DoubleBinaryOpCode::NOT_EQUAL, CODE> : std::not_equal_to<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
};
*/
template <LegateTypeCode CODE>
struct DoubleBinaryOp<DoubleBinaryOpCode::POWER, CODE> {
  using VAL                   = legate_type_of<CODE>;
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
  constexpr VAL operator()(const VAL& a, const VAL& b) const { return std::pow(a, b); }
};

template <>
struct DoubleBinaryOp<DoubleBinaryOpCode::POWER, LegateTypeCode::HALF_LT> {
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
  LEGATE_DEVICE_PREFIX __half operator()(const __half& a, const __half& b) const
  {
    return pow(a, b);
  }
};

template <>
struct DoubleBinaryOp<DoubleBinaryOpCode::POWER, LegateTypeCode::COMPLEX64_LT> {
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
  LEGATE_DEVICE_PREFIX complex<float> operator()(const complex<float>& a,
                                                 const complex<float>& b) const
  {
    return pow(a, b);
  }
};

template <>
struct DoubleBinaryOp<DoubleBinaryOpCode::POWER, LegateTypeCode::COMPLEX128_LT> {
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
  LEGATE_DEVICE_PREFIX complex<double> operator()(const complex<double>& a,
                                                  const complex<double>& b) const
  {
    return pow(a, b);
  }
};

template <LegateTypeCode CODE>
struct DoubleBinaryOp<DoubleBinaryOpCode::SUBTRACT, CODE> : std::minus<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args) {}
};
/*
template <LegateTypeCode CODE>
struct DoubleBinaryOp<DoubleBinaryOpCode::ALLCLOSE, CODE> {
  using VAL                   = legate_type_of<CODE>;
  static constexpr bool valid = true;

  DoubleBinaryOp() {}
  DoubleBinaryOp(const std::vector<UntypedScalar>& args)
  {
    assert(args.size() == 2);
    rtol_ = args[0].value<double>();
    atol_ = args[1].value<double>();
  }

  template <typename T = VAL, std::enable_if_t<!is_complex<T>::value>* = nullptr>
  constexpr bool operator()(const T& a, const T& b) const
  {
    using std::fabs;
    return fabs(static_cast<double>(a) - static_cast<double>(b)) <=
           atol_ + rtol_ * static_cast<double>(fabs(b));
  }

  template <typename T = VAL, std::enable_if_t<is_complex<T>::value>* = nullptr>
  constexpr bool operator()(const T& a, const T& b) const
  {
    return static_cast<double>(abs(a - b)) <= atol_ + rtol_ * static_cast<double>(abs(b));
  }

  double rtol_{0};
  double atol_{0};
};
*/
}  // namespace numpy
}  // namespace legate
