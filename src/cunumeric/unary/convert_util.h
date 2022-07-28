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
#include "cunumeric/isnan.h"

namespace cunumeric {

enum class ConvertCode : int {
  NOOP = CUNUMERIC_CONVERT_NAN_NOOP,
  PROD = CUNUMERIC_CONVERT_NAN_PROD,
  SUM  = CUNUMERIC_CONVERT_NAN_SUM,
};

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) op_dispatch(ConvertCode op_code, Functor f, Fnargs&&... args)
{
  switch (op_code) {
    case ScanCode::NOOPE:
      return f.template operator()<ConvertCode::NOOP>(std::forward<Fnargs>(args)...);
    case ScanCode::PROD:
      return f.template operator()<ConvertCode::PROD>(std::forward<Fnargs>(args)...);
    case ScanCode::SUM:
      return f.template operator()<ConvertCode::SUM>(std::forward<Fnargs>(args)...);
    default: break;
  }
  assert(false);
  return f.template operator()<ScanCode::NOOP, false>(std::forward<Fnargs>(args)...);
}

template <ConvertCode OP_CODE, legate::LegateTypeCode DST_TYPE, legate::LegateTypeCode SRC_TYPE>
struct ConvertOp {
};

template <legate::LegateTypeCode DST_TYPE, legate::LegateTypeCode SRC_TYPE>
struct ConvertOp<ConvertCode::NOOP, DST_TYPE, SRC_TYPE> {
  using SRC = legate::legate_type_of<SRC_TYPE>;
  using DST = legate::legate_type_of<DST_TYPE>;

  template <
    typename _SRC                                                                         = SRC,
    std::enable_if_t<!legate::is_complex<_SRC>::value or legate::is_complex<DST>::value>* = nullptr>
  constexpr DST operator()(const _SRC& src) const
  {
    return static_cast<DST>(src);
  }

  template <typename _SRC = SRC,
            std::enable_if_t<legate::is_complex<_SRC>::value and !legate::is_complex<DST>::value>* =
              nullptr>
  constexpr DST operator()(const _SRC& src) const
  {
    return static_cast<DST>(src.real());
  }
};

template <legate::LegateTypeCode SRC_TYPE>
struct ConvertOp<ConvertCode::NOOP, legate::LegateTypeCode::HALF_LT, SRC_TYPE> {
  using SRC = legate::legate_type_of<SRC_TYPE>;

  template <typename _SRC = SRC, std::enable_if_t<!legate::is_complex<_SRC>::value>* = nullptr>
  __CUDA_HD__ __half operator()(const _SRC& src) const
  {
    return static_cast<__half>(static_cast<double>(src));
  }

  template <typename _SRC = SRC, std::enable_if_t<legate::is_complex<_SRC>::value>* = nullptr>
  __CUDA_HD__ __half operator()(const _SRC& src) const
  {
    return static_cast<__half>(static_cast<double>(src.real()));
  }
};

template <legate::LegateTypeCode DST_TYPE>
struct ConvertOp<ConvertCode::NOOP, DST_TYPE, legate::LegateTypeCode::HALF_LT> {
  using DST = legate::legate_type_of<DST_TYPE>;

  constexpr DST operator()(const __half& src) const
  {
    return static_cast<DST>(static_cast<double>(src));
  }
};

template <legate::LegateTypeCode DST_TYPE, legate::LegateTypeCode SRC_TYPE>
struct ConvertOp<ConvertCode::PROD, DST_TYPE, SRC_TYPE> {
  using SRC = legate::legate_type_of<SRC_TYPE>;
  using DST = legate::legate_type_of<DST_TYPE>;

  cunumeric::isnan<SRC> isn;

  template <
    typename _SRC                                                                         = SRC,
    std::enable_if_t<!legate::is_complex<_SRC>::value or legate::is_complex<DST>::value>* = nullptr>
  constexpr DST operator()(const _SRC& src) const
  {
    return isn(src) ? static_cast<DST>(1) : static_cast<DST>(src);
  }

  template <typename _SRC = SRC,
            std::enable_if_t<legate::is_complex<_SRC>::value and !legate::is_complex<DST>::value>* =
              nullptr>
  constexpr DST operator()(const _SRC& src) const
  {
    return isn(src) ? static_cast<DST>(1) : static_cast<DST>(src.real());
  }
};

template <legate::LegateTypeCode SRC_TYPE>
struct ConvertOp<ConvertCode::PROD, legate::LegateTypeCode::HALF_LT, SRC_TYPE> {
  using SRC = legate::legate_type_of<SRC_TYPE>;

  cunumeric::isnan<SRC> isn;

  template <typename _SRC = SRC, std::enable_if_t<!legate::is_complex<_SRC>::value>* = nullptr>
  __CUDA_HD__ __half operator()(const _SRC& src) const
  {
    return isn(src) ? static_cast<__half>(1) : static_cast<__half>(static_cast<double>(src));
  }

  template <typename _SRC = SRC, std::enable_if_t<legate::is_complex<_SRC>::value>* = nullptr>
  __CUDA_HD__ __half operator()(const _SRC& src) const
  {
    return isn(src) ? static_cast<__half>(1) : static_cast<__half>(static_cast<double>(src.real()));
  }
};

template <legate::LegateTypeCode DST_TYPE>
struct ConvertOp<ConvertCode::PROD, DST_TYPE, legate::LegateTypeCode::HALF_LT> {
  using DST = legate::legate_type_of<DST_TYPE>;

  cunumeric::isnan<__half> isn;

  constexpr DST operator()(const __half& src) const
  {
    return isn(src) ? static_cast<DST>(1) : static_cast<DST>(static_cast<double>(src));
  }
};

template <legate::LegateTypeCode DST_TYPE, legate::LegateTypeCode SRC_TYPE>
struct ConvertOp<ConvertCode::SUM, DST_TYPE, SRC_TYPE> {
  using SRC = legate::legate_type_of<SRC_TYPE>;
  using DST = legate::legate_type_of<DST_TYPE>;

  cunumeric::isnan<SRC> isn;

  template <
    typename _SRC                                                                         = SRC,
    std::enable_if_t<!legate::is_complex<_SRC>::value or legate::is_complex<DST>::value>* = nullptr>
  constexpr DST operator()(const _SRC& src) const
  {
    return isn(src) ? static_cast<DST>(0) : static_cast<DST>(src);
  }

  template <typename _SRC = SRC,
            std::enable_if_t<legate::is_complex<_SRC>::value and !legate::is_complex<DST>::value>* =
              nullptr>
  constexpr DST operator()(const _SRC& src) const
  {
    return isn(src) ? static_cast<DST>(0) : static_cast<DST>(src.real());
  }
};

template <legate::LegateTypeCode SRC_TYPE>
struct ConvertOp<ConvertCode::SUM, legate::LegateTypeCode::HALF_LT, SRC_TYPE> {
  using SRC = legate::legate_type_of<SRC_TYPE>;

  cunumeric::isnan<SRC> isn;

  template <typename _SRC = SRC, std::enable_if_t<!legate::is_complex<_SRC>::value>* = nullptr>
  __CUDA_HD__ __half operator()(const _SRC& src) const
  {
    return isn(src) ? static_cast<__half>(0) : static_cast<__half>(static_cast<double>(src));
  }

  template <typename _SRC = SRC, std::enable_if_t<legate::is_complex<_SRC>::value>* = nullptr>
  __CUDA_HD__ __half operator()(const _SRC& src) const
  {
    return isn(src) ? static_cast<__half>(0) : static_cast<__half>(static_cast<double>(src.real()));
  }
};

template <legate::LegateTypeCode DST_TYPE>
struct ConvertOp<ConvertCode::SUM, DST_TYPE, legate::LegateTypeCode::HALF_LT> {
  using DST = legate::legate_type_of<DST_TYPE>;

  cunumeric::isnan<__half> isn;

  constexpr DST operator()(const __half& src) const
  {
    return isn(src) ? static_cast<DST>(0) : static_cast<DST>(static_cast<double>(src));
  }
};

}  // namespace cunumeric
