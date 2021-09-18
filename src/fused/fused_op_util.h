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

enum class FusedOpCode : int {
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
constexpr decltype(auto) op_dispatch(FusedOpCode op_code, Functor f, Fnargs&&... args)
{
  switch (op_code) {
    case FusedOpCode::ADD:
      return f.template operator()<FusedOpCode::ADD>(std::forward<Fnargs>(args)...);
    case FusedOpCode::DIVIDE:
      return f.template operator()<FusedOpCode::DIVIDE>(std::forward<Fnargs>(args)...);
    //case FusedOpCode::EQUAL:
    //  return f.template operator()<FusedOpCode::EQUAL>(std::forward<Fnargs>(args)...);
    case FusedOpCode::FLOOR_DIVIDE:
      return f.template operator()<FusedOpCode::FLOOR_DIVIDE>(std::forward<Fnargs>(args)...);
    //case FusedOpCode::GREATER:
    //  return f.template operator()<FusedOpCode::GREATER>(std::forward<Fnargs>(args)...);
    //case FusedOpCode::GREATER_EQUAL:
    //  return f.template operator()<FusedOpCode::GREATER_EQUAL>(std::forward<Fnargs>(args)...);
    //case FusedOpCode::LESS:
    //  return f.template operator()<FusedOpCode::LESS>(std::forward<Fnargs>(args)...);
    //case FusedOpCode::LESS_EQUAL:
    //  return f.template operator()<FusedOpCode::LESS_EQUAL>(std::forward<Fnargs>(args)...);
    case FusedOpCode::MAXIMUM:
      return f.template operator()<FusedOpCode::MAXIMUM>(std::forward<Fnargs>(args)...);
    case FusedOpCode::MINIMUM:
      return f.template operator()<FusedOpCode::MINIMUM>(std::forward<Fnargs>(args)...);
    case FusedOpCode::MOD:
      return f.template operator()<FusedOpCode::MOD>(std::forward<Fnargs>(args)...);
    case FusedOpCode::MULTIPLY:
      return f.template operator()<FusedOpCode::MULTIPLY>(std::forward<Fnargs>(args)...);
    //case FusedOpCode::NOT_EQUAL:
    //  return f.template operator()<FusedOpCode::NOT_EQUAL>(std::forward<Fnargs>(args)...);
    case FusedOpCode::POWER:
      return f.template operator()<FusedOpCode::POWER>(std::forward<Fnargs>(args)...);
    case FusedOpCode::SUBTRACT:
      return f.template operator()<FusedOpCode::SUBTRACT>(std::forward<Fnargs>(args)...);
    default: break;
  }
  assert(false);
  return f.template operator()<FusedOpCode::ADD>(std::forward<Fnargs>(args)...);
}

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) reduce_op_dispatch(FusedOpCode op_code, Functor f, Fnargs&&... args)
{
  switch (op_code) {
    //case FusedOpCode::EQUAL:
    //  return f.template operator()<FusedOpCode::EQUAL>(std::forward<Fnargs>(args)...);
    //case FusedOpCode::ALLCLOSE:
    //  return f.template operator()<FusedOpCode::ALLCLOSE>(std::forward<Fnargs>(args)...);
    default: break;
  }
  assert(false);
  //return f.template operator()<FusedOpCode::EQUAL>(std::forward<Fnargs>(args)...);
}

template <FusedOpCode OP_CODE, LegateTypeCode CODE>
struct FusedOp {
  static constexpr bool valid = false;
};

template <LegateTypeCode CODE>
struct FusedOp<FusedOpCode::ADD, CODE> : std::plus<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
};

template <LegateTypeCode CODE>
struct FusedOp<FusedOpCode::DIVIDE, CODE> : std::divides<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
};
/*
template <LegateTypeCode CODE>
struct FusedOp<FusedOpCode::EQUAL, CODE> : std::equal_to<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
};
*/
using std::floor;
template <LegateTypeCode CODE>
struct FusedOp<FusedOpCode::FLOOR_DIVIDE, CODE> {
  using T                     = legate_type_of<CODE>;
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
  constexpr T operator()(const T& a, const T& b) const { return floor(a / b); }
};

template <>
struct FusedOp<FusedOpCode::FLOOR_DIVIDE, LegateTypeCode::COMPLEX64_LT> {
  static constexpr bool valid = false;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
};

template <>
struct FusedOp<FusedOpCode::FLOOR_DIVIDE, LegateTypeCode::COMPLEX128_LT> {
  static constexpr bool valid = false;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
};
/*
template <LegateTypeCode CODE>
struct FusedOp<FusedOpCode::GREATER, CODE> : std::greater<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
};

template <LegateTypeCode CODE>
struct FusedOp<FusedOpCode::GREATER_EQUAL, CODE> : std::greater_equal<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
};

template <LegateTypeCode CODE>
struct FusedOp<FusedOpCode::LESS, CODE> : std::less<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
};

template <LegateTypeCode CODE>
struct FusedOp<FusedOpCode::LESS_EQUAL, CODE> : std::less_equal<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
};
*/
template <LegateTypeCode CODE>
struct FusedOp<FusedOpCode::MAXIMUM, CODE> {
  using T                     = legate_type_of<CODE>;
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
  constexpr T operator()(const T& a, const T& b) const { return std::max<T>(a, b); }
};

template <LegateTypeCode CODE>
struct FusedOp<FusedOpCode::MINIMUM, CODE> {
  using T                     = legate_type_of<CODE>;
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
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
struct FusedOp<FusedOpCode::MOD, CODE> {
  using T                     = legate_type_of<CODE>;
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
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
struct FusedOp<FusedOpCode::MOD, LegateTypeCode::HALF_LT> {
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
  LEGATE_DEVICE_PREFIX __half operator()(const __half& a, const __half& b) const
  {
    return static_cast<__half>(real_mod(static_cast<float>(a), static_cast<float>(b)));
  }
};

template <>
struct FusedOp<FusedOpCode::MOD, LegateTypeCode::COMPLEX64_LT> {
  static constexpr bool valid = false;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
};

template <>
struct FusedOp<FusedOpCode::MOD, LegateTypeCode::COMPLEX128_LT> {
  static constexpr bool valid = false;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
};

template <LegateTypeCode CODE>
struct FusedOp<FusedOpCode::MULTIPLY, CODE> : std::multiplies<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
};
/*
template <LegateTypeCode CODE>
struct FusedOp<FusedOpCode::NOT_EQUAL, CODE> : std::not_equal_to<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
};
*/
template <LegateTypeCode CODE>
struct FusedOp<FusedOpCode::POWER, CODE> {
  using VAL                   = legate_type_of<CODE>;
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
  constexpr VAL operator()(const VAL& a, const VAL& b) const { return std::pow(a, b); }
};

template <>
struct FusedOp<FusedOpCode::POWER, LegateTypeCode::HALF_LT> {
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
  LEGATE_DEVICE_PREFIX __half operator()(const __half& a, const __half& b) const
  {
    return pow(a, b);
  }
};

template <>
struct FusedOp<FusedOpCode::POWER, LegateTypeCode::COMPLEX64_LT> {
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
  LEGATE_DEVICE_PREFIX complex<float> operator()(const complex<float>& a,
                                                 const complex<float>& b) const
  {
    return pow(a, b);
  }
};

template <>
struct FusedOp<FusedOpCode::POWER, LegateTypeCode::COMPLEX128_LT> {
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
  LEGATE_DEVICE_PREFIX complex<double> operator()(const complex<double>& a,
                                                  const complex<double>& b) const
  {
    return pow(a, b);
  }
};

template <LegateTypeCode CODE>
struct FusedOp<FusedOpCode::SUBTRACT, CODE> : std::minus<legate_type_of<CODE>> {
  static constexpr bool valid = true;
  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args) {}
};
/*
template <LegateTypeCode CODE>
struct FusedOp<FusedOpCode::ALLCLOSE, CODE> {
  using VAL                   = legate_type_of<CODE>;
  static constexpr bool valid = true;

  FusedOp() {}
  FusedOp(const std::vector<UntypedScalar>& args)
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
