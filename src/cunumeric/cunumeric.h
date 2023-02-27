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

#include "legate.h"
#include "cunumeric/cunumeric_c.h"

namespace cunumeric {

using Array = legate::Store;

enum class VariantKind : int {
  CPU = 0,
  OMP = 1,
  GPU = 2,
};

struct CuNumeric {
 public:
  template <typename... Args>
  static void record_variant(Args&&... args)
  {
    get_registrar().record_variant(std::forward<Args>(args)...);
  }
  static legate::TaskRegistrar& get_registrar();
};

template <typename T>
struct CuNumericTask : public legate::LegateTask<T> {
  using Registrar = CuNumeric;
};

}  // namespace cunumeric
