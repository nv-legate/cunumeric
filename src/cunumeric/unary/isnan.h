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

#include "legion.h"

namespace cunumeric {

template <typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
constexpr bool is_nan(const T& x)
{
  return false;
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
__CUDA_HD__ bool is_nan(const T& x)
{
  return std::isnan(x);
}

template <typename T>
__CUDA_HD__ bool is_nan(const complex<T>& x)
{
  return std::isnan(x.imag()) || std::isnan(x.real());
}

__CUDA_HD__ inline bool is_nan(const __half& x) { return isnan(x); }

}  // namespace cunumeric
