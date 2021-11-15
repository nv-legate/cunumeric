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
#include "cunumeric/unary/unary_red_util.h"

namespace cunumeric {

struct ScalarUnaryRedArgs {
  const Array& out;
  const Array& in;
  UnaryRedCode op_code;
  std::vector<legate::Store> args;
};

// Unary reduction task that produces scalar results
class ScalarUnaryRedTask : public CuNumericTask<ScalarUnaryRedTask> {
 public:
  static const int TASK_ID = CUNUMERIC_SCALAR_UNARY_RED;

 public:
  static void cpu_variant(legate::TaskContext& context);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& context);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

namespace detail {
template <typename _T, std::enable_if_t<!legate::is_complex<_T>::value>* = nullptr>
__host__ __device__ inline bool convert_to_bool(const _T& in)
{
  return bool(in);
}
template <typename _T, std::enable_if_t<legate::is_complex<_T>::value>* = nullptr>
__host__ __device__ inline bool convert_to_bool(const _T& in)
{
  return bool(in.real());
}
}

}  // namespace cunumeric
