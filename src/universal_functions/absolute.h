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

#ifndef __NUMPY_ABSOLUTE_H__
#define __NUMPY_ABSOLUTE_H__

#include "universal_function.h"
#include <cmath>

namespace legate {
namespace numpy {
using std::abs;
template<class T>
struct AbsoluteOperation {
  using argument_type           = T;
  constexpr static auto op_code = NumPyOpCode::NUMPY_ABSOLUTE;

  template<class _T = T, std::enable_if_t<std::is_integral<_T>::value and std::is_signed<_T>::value>* = nullptr>
  constexpr T operator()(const T& x) const {
    return abs(x);
  }

  template<class _T = T, std::enable_if_t<std::is_integral<_T>::value and std::is_unsigned<_T>::value>* = nullptr>
  constexpr T operator()(const T& x) const {
    return x;
  }

  template<class _T = T, std::enable_if_t<!std::is_integral<_T>::value>* = nullptr>
  constexpr auto operator()(const T& x) const {
    using std::fabs;
    return fabs(x);
  }
};

template<typename T>
using Absolute = UnaryUniversalFunction<AbsoluteOperation<T>>;

}    // namespace numpy
}    // namespace legate

#endif    // __NUMPY_ABSOLUTE_H__
