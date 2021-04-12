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

#ifndef __NUMPY_POWER_H__
#define __NUMPY_POWER_H__

#include "universal_function.h"
#include <functional>

namespace legate {
namespace numpy {
using std::pow;
template<class T>
struct PowerOperation {
  using first_argument_type     = T;
  using second_argument_type    = T;
  constexpr static auto op_code = NumPyOpCode::NUMPY_POWER;

  __CUDA_HD__ constexpr T operator()(const T& base, const T& exponent) const { return T(pow(base, exponent)); }
};

template<typename T>
using Power = NoncommutativeBinaryUniversalFunction<PowerOperation<T>>;
}    // namespace numpy
}    // namespace legate

#endif    // __NUMPY_POWER_H__
