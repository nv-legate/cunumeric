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

#ifndef __NUMPY_CEIL_H__
#define __NUMPY_CEIL_H__

#include "universal_function.h"
#include <cmath>

namespace legate {
namespace numpy {
using std::ceil;
template<class T>
struct CeilOperation {
  using argument_type           = T;
  constexpr static auto op_code = NumPyOpCode::NUMPY_CEIL;

  constexpr T operator()(const T& a) const { return ceil(a); }
};

template<typename T>
using Ceil = UnaryUniversalFunction<CeilOperation<T>>;
}    // namespace numpy
}    // namespace legate

#endif    // __NUMPY_CEIL_H__
