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

#include "cunumeric/cunumeric.h"

namespace cunumeric {

using namespace legate;

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) fft_dispatch(CuNumericFFTType type, Functor f, Fnargs&&... args)
{
  switch (type) {
    case CUNUMERIC_FFT_R2C:
      return f.template operator()<CUNUMERIC_FFT_R2C>(std::forward<Fnargs>(args)...);
    case CUNUMERIC_FFT_C2R:
      return f.template operator()<CUNUMERIC_FFT_C2R>(std::forward<Fnargs>(args)...);
    case CUNUMERIC_FFT_C2C:
      return f.template operator()<CUNUMERIC_FFT_C2C>(std::forward<Fnargs>(args)...);
    case CUNUMERIC_FFT_D2Z:
      return f.template operator()<CUNUMERIC_FFT_D2Z>(std::forward<Fnargs>(args)...);
    case CUNUMERIC_FFT_Z2D:
      return f.template operator()<CUNUMERIC_FFT_Z2D>(std::forward<Fnargs>(args)...);
    case CUNUMERIC_FFT_Z2Z:
      return f.template operator()<CUNUMERIC_FFT_Z2Z>(std::forward<Fnargs>(args)...);
    default: break;
  }
  assert(false);
  return f.template operator()<CUNUMERIC_FFT_C2C>(std::forward<Fnargs>(args)...);
}

template <CuNumericFFTType TYPE, Type CODE_IN>
struct FFT {
  static constexpr bool valid = false;
};

template <>
struct FFT<CUNUMERIC_FFT_R2C, Type::FLOAT32> {
  static constexpr bool valid    = true;
  static constexpr Type CODE_OUT = Type::COMPLEX64;
};

template <>
struct FFT<CUNUMERIC_FFT_C2R, Type::COMPLEX64> {
  static constexpr bool valid    = true;
  static constexpr Type CODE_OUT = Type::FLOAT32;
};

template <>
struct FFT<CUNUMERIC_FFT_C2C, Type::COMPLEX64> {
  static constexpr bool valid    = true;
  static constexpr Type CODE_OUT = Type::COMPLEX64;
};

template <>
struct FFT<CUNUMERIC_FFT_D2Z, Type::FLOAT64> {
  static constexpr bool valid    = true;
  static constexpr Type CODE_OUT = Type::COMPLEX128;
};

template <>
struct FFT<CUNUMERIC_FFT_Z2D, Type::COMPLEX128> {
  static constexpr bool valid    = true;
  static constexpr Type CODE_OUT = Type::FLOAT64;
};

template <>
struct FFT<CUNUMERIC_FFT_Z2Z, Type::COMPLEX128> {
  static constexpr bool valid    = true;
  static constexpr Type CODE_OUT = Type::COMPLEX128;
};

}  // namespace cunumeric
