/* Copyright 2022 NVIDIA Corporation
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

#include "cunumeric/cunumeric_c.h"

namespace cunumeric {

template <CuNumericTypeCodes CODE>
struct CuNumericTypeOf {
  using type = Legion::Point<1>;
};
#if LEGION_MAX_DIM >= 1
template <>
struct CuNumericTypeOf<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT1> {
  using type = Legion::Point<1>;
};
#endif
#if LEGION_MAX_DIM >= 2
template <>
struct CuNumericTypeOf<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT2> {
  using type = Legion::Point<2>;
};
#endif
#if LEGION_MAX_DIM >= 3
template <>
struct CuNumericTypeOf<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT3> {
  using type = Legion::Point<3>;
};
#endif
#if LEGION_MAX_DIM >= 4
template <>
struct CuNumericTypeOf<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT4> {
  using type = Legion::Point<4>;
};
#endif
#if LEGION_MAX_DIM >= 5
template <>
struct CuNumericTypeOf<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT4> {
  using type = Legion::Point<5>;
};
#endif
#if LEGION_MAX_DIM >= 6
template <>
struct CuNumericTypeOf<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT4> {
  using type = Legion::Point<6>;
};
#endif
#if LEGION_MAX_DIM >= 7
template <>
struct CuNumericTypeOf<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT4> {
  using type = Legion::Point<7>;
};
#endif
#if LEGION_MAX_DIM >= 8
template <>
struct CuNumericTypeOf<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT4> {
  using type = Legion::Point<8>;
};
#endif
#if LEGION_MAX_DIM >= 9
template <>
struct CuNumericTypeOf<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT4> {
  using type = Legion::Point<9>;
};
#endif

template <CuNumericTypeCodes CODE>
using cunumeric_type_of = typename CuNumericTypeOf<CODE>::type;

}  // namespace cunumeric
