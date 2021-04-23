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

#ifndef __NUMPY_SQRT_H__
#define __NUMPY_SQRT_H__

#include "universal_function.h"
#include <cmath>

namespace legate {
namespace numpy {
using std::sqrt;
template <class T>
struct SqrtOperation {
  using argument_type           = T;
  using result_type             = decltype(sqrt(std::declval<argument_type>()));
  constexpr static auto op_code = NumPyOpCode::NUMPY_SQRT;

  __CUDA_HD__ constexpr result_type operator()(const argument_type& a) const { return sqrt(a); }
};

template <typename T>
using Sqrt = UnaryUniversalFunction<SqrtOperation<T>>;
}  // namespace numpy
}  // namespace legate

#endif  // __NUMPY_SQRT_H__
