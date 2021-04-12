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

#ifndef __NUMPY_LOGICAL_NOT_H__
#define __NUMPY_LOGICAL_NOT_H__

#include "universal_function.h"
#include <functional>

namespace legate {
namespace numpy {
template<class T>
struct LogicalNotOperation : public std::logical_not<T> {
  using argument_type           = typename std::logical_not<T>::argument_type;
  using result_type             = typename std::logical_not<T>::result_type;
  constexpr static auto op_code = NumPyOpCode::NUMPY_LOGICAL_NOT;

  template<typename T_>
  constexpr bool operator()(const T_& x) const {
    return std::logical_not<T_>{}(x);
  }

  template<typename T_>
  constexpr bool operator()(const complex<T_>& x) const {
    return std::logical_not<T_>{}(x.real()) && std::logical_not<T_>{}(x.imag());
  }
};

template<typename T>
using LogicalNot = UnaryUniversalFunction<LogicalNotOperation<T>>;
}    // namespace numpy
}    // namespace legate

#endif    // __NUMPY_LOGICAL_NOT_H__
