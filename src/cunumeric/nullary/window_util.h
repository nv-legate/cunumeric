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

enum class WindowOpCode : int {
  BARLETT  = CUNUMERIC_WINDOW_BARLETT,
  BLACKMAN = CUNUMERIC_WINDOW_BLACKMAN,
  HAMMING  = CUNUMERIC_WINDOW_HAMMING,
  HANNING  = CUNUMERIC_WINDOW_HANNING,
  KAISER   = CUNUMERIC_WINDOW_KAISER,
};

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) op_dispatch(WindowOpCode op_code, Functor f, Fnargs&&... args)
{
  switch (op_code) {
    case WindowOpCode::BARLETT:
      return f.template operator()<WindowOpCode::BARLETT>(std::forward<Fnargs>(args)...);
    case WindowOpCode::BLACKMAN:
      return f.template operator()<WindowOpCode::BLACKMAN>(std::forward<Fnargs>(args)...);
    case WindowOpCode::HAMMING:
      return f.template operator()<WindowOpCode::HAMMING>(std::forward<Fnargs>(args)...);
    case WindowOpCode::HANNING:
      return f.template operator()<WindowOpCode::HANNING>(std::forward<Fnargs>(args)...);
    case WindowOpCode::KAISER:
      return f.template operator()<WindowOpCode::KAISER>(std::forward<Fnargs>(args)...);
  }
  assert(false);
  return f.template operator()<WindowOpCode::BARLETT>(std::forward<Fnargs>(args)...);
}

template <WindowOpCode OP_CODE>
struct WindowOp;

template <>
struct WindowOp<WindowOpCode::BARLETT> {
  WindowOp(const std::vector<legate::Scalar>& scalars) {}
  constexpr double operator()(int64_t idx) { return 0.0; }
};

template <>
struct WindowOp<WindowOpCode::BLACKMAN> {
  WindowOp(const std::vector<legate::Scalar>& scalars) {}
  constexpr double operator()(int64_t idx) { return 0.0; }
};

template <>
struct WindowOp<WindowOpCode::HAMMING> {
  WindowOp(const std::vector<legate::Scalar>& scalars) {}
  constexpr double operator()(int64_t idx) { return 0.0; }
};

template <>
struct WindowOp<WindowOpCode::HANNING> {
  WindowOp(const std::vector<legate::Scalar>& scalars) {}
  constexpr double operator()(int64_t idx) { return 0.0; }
};

template <>
struct WindowOp<WindowOpCode::KAISER> {
  WindowOp(const std::vector<legate::Scalar>& scalars) : beta_(scalars.back().value<double>()) {}

  constexpr double operator()(int64_t idx) { return 0.0; }

  double beta_;
};

}  // namespace cunumeric
