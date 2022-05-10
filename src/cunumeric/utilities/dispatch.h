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

#include "legate.h"
#include "cunumeric/cunumeric_c.h"

namespace cunumeric {

template <int DIM>
struct inner_type_dispatch_fn {
  template <typename TypeCode, typename Functor, typename... Fnargs>
  constexpr decltype(auto) operator()(TypeCode code, Functor f, Fnargs&&... args)
  {
    switch (code) {
      case CuNumericTypeCodes::CUNUMERIC_TYPE_POINT1: {
        return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT1, DIM>(
          std::forward<Fnargs>(args)...);
      }
      case CuNumericTypeCodes::CUNUMERIC_TYPE_POINT2: {
        return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT2, DIM>(
          std::forward<Fnargs>(args)...);
      }
      case CuNumericTypeCodes::CUNUMERIC_TYPE_POINT3: {
        return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT3, DIM>(
          std::forward<Fnargs>(args)...);
      }
      case CuNumericTypeCodes::CUNUMERIC_TYPE_POINT4: {
        return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT4, DIM>(
          std::forward<Fnargs>(args)...);
      }
      case CuNumericTypeCodes::CUNUMERIC_TYPE_POINT5: {
        return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT5, DIM>(
          std::forward<Fnargs>(args)...);
      }
      case CuNumericTypeCodes::CUNUMERIC_TYPE_POINT6: {
        return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT6, DIM>(
          std::forward<Fnargs>(args)...);
      }
      case CuNumericTypeCodes::CUNUMERIC_TYPE_POINT7: {
        return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT7, DIM>(
          std::forward<Fnargs>(args)...);
      }
      case CuNumericTypeCodes::CUNUMERIC_TYPE_POINT8: {
        return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT8, DIM>(
          std::forward<Fnargs>(args)...);
      }
      case CuNumericTypeCodes::CUNUMERIC_TYPE_POINT9: {
        return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT9, DIM>(
          std::forward<Fnargs>(args)...);
      }
      default: {
        return legate::inner_type_dispatch_fn<DIM>{}(code, f, std::forward<Fnargs>(args)...);
      }
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
      return inner_type_dispatch_fn<2>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGION_MAX_DIM >= 3
    case 3: {
      return inner_type_dispatch_fn<3>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGION_MAX_DIM >= 4
    case 4: {
      return inner_type_dispatch_fn<4>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGION_MAX_DIM >= 5
    case 5: {
      return inner_type_dispatch_fn<5>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGION_MAX_DIM >= 6
    case 6: {
      return inner_type_dispatch_fn<6>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGION_MAX_DIM >= 7
    case 7: {
      return inner_type_dispatch_fn<7>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGION_MAX_DIM >= 8
    case 8: {
      return inner_type_dispatch_fn<8>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGION_MAX_DIM >= 9
    case 9: {
      return inner_type_dispatch_fn<9>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
  }
  assert(false);
  return inner_type_dispatch_fn<1>{}(code, f, std::forward<Fnargs>(args)...);
}

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) double_dispatch(int dim1, int dim2, Functor f, Fnargs&&... args)
{
  return legate::double_dispatch(
    dim1, dim2, std::forward<Functor>(f), std::forward<Fnargs>(args)...);
}

// template <typename Functor, typename... Fnargs>
// constexpr decltype(auto) dim_dispatch(int dim, Functor f, Fnargs&&... args)
//{
//   return legate::dim_dispatch(dim, std::forward<Functor>(f),std::forward<Fnargs>(args)...);
// }

template <typename TypeCode, typename Functor, typename... Fnargs>
constexpr decltype(auto) type_dispatch(TypeCode code, Functor f, Fnargs&&... args)
{
  switch (code) {
    case CuNumericTypeCodes::CUNUMERIC_TYPE_POINT1: {
      return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT1>(
        std::forward<Fnargs>(args)...);
    }
    case CuNumericTypeCodes::CUNUMERIC_TYPE_POINT2: {
      return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT2>(
        std::forward<Fnargs>(args)...);
    }
    case CuNumericTypeCodes::CUNUMERIC_TYPE_POINT3: {
      return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT3>(
        std::forward<Fnargs>(args)...);
    }
    case CuNumericTypeCodes::CUNUMERIC_TYPE_POINT4: {
      return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT4>(
        std::forward<Fnargs>(args)...);
    }
    case CuNumericTypeCodes::CUNUMERIC_TYPE_POINT5: {
      return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT5>(
        std::forward<Fnargs>(args)...);
    }
    case CuNumericTypeCodes::CUNUMERIC_TYPE_POINT6: {
      return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT6>(
        std::forward<Fnargs>(args)...);
    }
    case CuNumericTypeCodes::CUNUMERIC_TYPE_POINT7: {
      return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT7>(
        std::forward<Fnargs>(args)...);
    }
    case CuNumericTypeCodes::CUNUMERIC_TYPE_POINT8: {
      return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT8>(
        std::forward<Fnargs>(args)...);
    }
    case CuNumericTypeCodes::CUNUMERIC_TYPE_POINT9: {
      return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT9>(
        std::forward<Fnargs>(args)...);
    }
    default: {
      legate::type_dispatch(code, std::forward<Functor>(f), std::forward<Fnargs>(args)...);
    }
  }
  assert(false);
  return f.template operator()<CuNumericTypeCodes::CUNUMERIC_TYPE_POINT1>(
    std::forward<Fnargs>(args)...);
}

}  // namespace cunumeric
