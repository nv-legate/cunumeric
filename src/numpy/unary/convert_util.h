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

#include "numpy/numpy.h"

namespace legate {
namespace numpy {

template <LegateTypeCode DST_TYPE, LegateTypeCode SRC_TYPE>
struct ConvertOp {
  using SRC = legate_type_of<SRC_TYPE>;
  using DST = legate_type_of<DST_TYPE>;

  template <typename _SRC = SRC, std::enable_if_t<!is_complex<_SRC>::value>* = nullptr>
  constexpr DST operator()(const _SRC& src) const
  {
    return static_cast<DST>(src);
  }

  template <typename _SRC = SRC, std::enable_if_t<is_complex<_SRC>::value>* = nullptr>
  constexpr DST operator()(const _SRC& src) const
  {
    return static_cast<DST>(src.real());
  }
};

template <LegateTypeCode SRC_TYPE>
struct ConvertOp<LegateTypeCode::HALF_LT, SRC_TYPE> {
  using SRC = legate_type_of<SRC_TYPE>;

  template <typename _SRC = SRC, std::enable_if_t<!is_complex<_SRC>::value>* = nullptr>
  __CUDA_HD__ __half operator()(const _SRC& src) const
  {
    return static_cast<__half>(static_cast<double>(src));
  }

  template <typename _SRC = SRC, std::enable_if_t<is_complex<_SRC>::value>* = nullptr>
  __CUDA_HD__ __half operator()(const _SRC& src) const
  {
    return static_cast<__half>(static_cast<double>(src.real()));
  }
};

template <LegateTypeCode DST_TYPE>
struct ConvertOp<DST_TYPE, LegateTypeCode::HALF_LT> {
  using DST = legate_type_of<DST_TYPE>;

  constexpr DST operator()(const __half& src) const
  {
    return static_cast<DST>(static_cast<double>(src));
  }
};

}  // namespace numpy
}  // namespace legate
