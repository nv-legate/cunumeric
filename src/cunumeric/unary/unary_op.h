/* Copyright 2021-2022 NVIDIA Corporation
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
#include "cunumeric/unary/unary_op_util.h"

namespace cunumeric {

struct UnaryOpArgs {
  const Array& in;
  const Array& out;
  UnaryOpCode op_code;
  std::vector<legate::Store> args;
};

struct MultiOutUnaryOpArgs {
  const Array& in;
  const Array& out1;
  const Array& out2;
  UnaryOpCode op_code;
};

class UnaryOpTask : public CuNumericTask<UnaryOpTask> {
 public:
  static const int TASK_ID = CUNUMERIC_UNARY_OP;

 public:
  static void cpu_variant(legate::TaskContext& context);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& context);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

template <int DIM>
struct inner_type_dispatch_fn {
  template <typename TypeCode, typename Functor, typename... Fnargs>
  constexpr decltype(auto) operator()(TypeCode c, Functor f, Fnargs&&... args)
  {
    if (c <= MAX_TYPE_NUMBER) {
      return legate::inner_type_dispatch_fn<DIM>{}(c, f, std::forward<Fnargs>(args)...);
    }
    size_t code = static_cast<size_t>(c);
    switch (code) {
#if LEGION_MAX_DIM >= 1
      case MAX_TYPE_NUMBER + 1: {  // CuNumericTypeCodes::CUNUMERIC_TYPE_POINT1:
        return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT1, DIM>(
          std::forward<Fnargs>(args)...);
      }
#endif
#if LEGION_MAX_DIM >= 2
      case MAX_TYPE_NUMBER + 2: {  // CuNumericTypeCodes::CUNUMERIC_TYPE_POINT2
        return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT2, DIM>(
          std::forward<Fnargs>(args)...);
      }
#endif
#if LEGION_MAX_DIM >= 3
      case MAX_TYPE_NUMBER + 3: {  // CuNumericTypeCodes::CUNUMERIC_TYPE_POINT3
        return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT3, DIM>(
          std::forward<Fnargs>(args)...);
      }
#endif
#if LEGION_MAX_DIM >= 4
      case MAX_TYPE_NUMBER + 4: {  // CuNumericTypeCodes::CUNUMERIC_TYPE_POINT4
        return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT4, DIM>(
          std::forward<Fnargs>(args)...);
      }
#endif
#if LEGION_MAX_DIM >= 5
      case MAX_TYPE_NUMBER + 5: {  // CuNumericTypeCodes::CUNUMERIC_TYPE_POINT5
        return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT5, DIM>(
          std::forward<Fnargs>(args)...);
      }
#endif
#if LEGION_MAX_DIM >= 6
      case MAX_TYPE_NUMBER + 6: {  // CuNumericTypeCodes::CUNUMERIC_TYPE_POINT6
        return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT6, DIM>(
          std::forward<Fnargs>(args)...);
      }
#endif
#if LEGION_MAX_DIM >= 7
      case MAX_TYPE_NUMBER + 7: {  // CuNumericTypeCodes::CUNUMERIC_TYPE_POINT7
        return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT7, DIM>(
          std::forward<Fnargs>(args)...);
      }
#endif
#if LEGION_MAX_DIM >= 8
      case MAX_TYPE_NUMBER + 8: {  // CuNumericTypeCodes::CUNUMERIC_TYPE_POINT8
        return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT8, DIM>(
          std::forward<Fnargs>(args)...);
      }
#endif
#if LEGION_MAX_DIM >= 9
      case MAX_TYPE_NUMBER + 9: {  // CuNumericTypeCodes::CUNUMERIC_TYPE_POINT9
        return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT9, DIM>(
          std::forward<Fnargs>(args)...);
      }
      default: break;
#endif
    }
    assert(false);
    return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT1, DIM>(
      std::forward<Fnargs>(args)...);
  }
};

template <typename TypeCode, typename Functor, typename... Fnargs>
constexpr decltype(auto) double_dispatch(int dim, TypeCode code, Functor f, Fnargs&&... args)
{
  switch (dim) {
#if LEGION_MAX_DIM >= 1
    case 1: {
      return cunumeric::inner_type_dispatch_fn<1>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGION_MAX_DIM >= 2
    case 2: {
      return cunumeric::inner_type_dispatch_fn<2>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGION_MAX_DIM >= 3
    case 3: {
      return cunumeric::inner_type_dispatch_fn<3>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGION_MAX_DIM >= 4
    case 4: {
      return cunumeric::inner_type_dispatch_fn<4>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGION_MAX_DIM >= 5
    case 5: {
      return cunumeric::inner_type_dispatch_fn<5>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGION_MAX_DIM >= 6
    case 6: {
      return cunumeric::inner_type_dispatch_fn<6>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGION_MAX_DIM >= 7
    case 7: {
      return cunumeric::inner_type_dispatch_fn<7>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGION_MAX_DIM >= 8
    case 8: {
      return cunumeric::inner_type_dispatch_fn<8>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGION_MAX_DIM >= 9
    case 9: {
      return cunumeric::inner_type_dispatch_fn<9>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
  }
  assert(false);
  return cunumeric::inner_type_dispatch_fn<1>{}(code, f, std::forward<Fnargs>(args)...);
}

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
struct CuNumericTypeOf<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT5> {
  using type = Legion::Point<5>;
};
#endif
#if LEGION_MAX_DIM >= 6
template <>
struct CuNumericTypeOf<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT6> {
  using type = Legion::Point<6>;
};
#endif
#if LEGION_MAX_DIM >= 7
template <>
struct CuNumericTypeOf<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT7> {
  using type = Legion::Point<7>;
};
#endif
#if LEGION_MAX_DIM >= 8
template <>
struct CuNumericTypeOf<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT8> {
  using type = Legion::Point<8>;
};
#endif
#if LEGION_MAX_DIM >= 9
template <>
struct CuNumericTypeOf<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT9> {
  using type = Legion::Point<9>;
};
#endif

template <CuNumericTypeCodes CODE>
using cunumeric_type_of = typename CuNumericTypeOf<CODE>::type;

}  // namespace cunumeric
