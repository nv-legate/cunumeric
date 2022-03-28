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

#include "cunumeric/cunumeric.h"

namespace cunumeric {

using namespace legate;

// These fft types match cufftType
enum class fftType {
    FFT_R2C = 0x2a,  // Real to complex (interleaved) 
    FFT_C2R = 0x2c,  // Complex (interleaved) to real 
    FFT_C2C = 0x29,  // Complex to complex (interleaved) 
    FFT_D2Z = 0x6a,  // Double to double-complex (interleaved) 
    FFT_Z2D = 0x6c,  // Double-complex (interleaved) to double 
    FFT_Z2Z = 0x69   // Double-complex to double-complex (interleaved)
};

enum class fftDirection {
  FFT_FORWARD = -1,
  FFT_INVERSE =  1

};

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) fft_dispatch(fftType type, Functor f, Fnargs&&... args)
{
  switch (type) {
    case fftType::FFT_R2C:
      return f.template operator()<fftType::FFT_R2C>(std::forward<Fnargs>(args)...);
    case fftType::FFT_C2R:
      return f.template operator()<fftType::FFT_C2R>(std::forward<Fnargs>(args)...);
    case fftType::FFT_C2C:
      return f.template operator()<fftType::FFT_C2C>(std::forward<Fnargs>(args)...);
    case fftType::FFT_D2Z:
      return f.template operator()<fftType::FFT_D2Z>(std::forward<Fnargs>(args)...);
    case fftType::FFT_Z2D:
      return f.template operator()<fftType::FFT_Z2D>(std::forward<Fnargs>(args)...);
    case fftType::FFT_Z2Z:
      return f.template operator()<fftType::FFT_Z2Z>(std::forward<Fnargs>(args)...);
    default: break;
  }
  assert(false);
  return f.template operator()<fftType::FFT_C2C>(std::forward<Fnargs>(args)...);
}

template <fftType TYPE, LegateTypeCode CODE_IN>
struct FFT {
  static constexpr bool valid = false;
};

template <>
struct FFT<fftType::FFT_R2C, LegateTypeCode::FLOAT_LT> {
  static constexpr bool valid = true;
  static constexpr LegateTypeCode CODE_OUT = LegateTypeCode::COMPLEX64_LT;
};

template <>
struct FFT<fftType::FFT_C2R, LegateTypeCode::COMPLEX64_LT> {
  static constexpr bool valid = true;
  static constexpr LegateTypeCode CODE_OUT = LegateTypeCode::FLOAT_LT;
};

template <>
struct FFT<fftType::FFT_C2C, LegateTypeCode::COMPLEX64_LT> {
  static constexpr bool valid = true;
  static constexpr LegateTypeCode CODE_OUT = LegateTypeCode::COMPLEX64_LT;
};

template <>
struct FFT<fftType::FFT_D2Z, LegateTypeCode::DOUBLE_LT> {
  static constexpr bool valid = true;
  static constexpr LegateTypeCode CODE_OUT = LegateTypeCode::COMPLEX128_LT;
};

template <>
struct FFT<fftType::FFT_Z2D, LegateTypeCode::COMPLEX128_LT> {
  static constexpr bool valid = true;
  static constexpr LegateTypeCode CODE_OUT = LegateTypeCode::DOUBLE_LT;
};

template <>
struct FFT<fftType::FFT_Z2Z, LegateTypeCode::COMPLEX128_LT> {
  static constexpr bool valid = true;
  static constexpr LegateTypeCode CODE_OUT = LegateTypeCode::COMPLEX128_LT;
};


}  // namespace cunumeric
