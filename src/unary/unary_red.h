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

#include "numpy.h"
#include "unary/unary_red_util.h"

namespace legate {
namespace numpy {

struct UnaryRedArgs {
  const Array &lhs;
  const Array &rhs;
  int32_t collapsed_dim;
  UnaryRedCode op_code;
};

class UnaryRedTask : public NumPyTask<UnaryRedTask> {
 public:
  static const int TASK_ID = NUMPY_UNARY_RED;
  // TODO: The first region requirement of this task can have either
  //       write discard or reduction privilege. For now we don't
  //       count that requirement just to avoid specifying a normal SOA
  //       layout constraint for it. There are two proper fixes for this:
  //       1) we attach each variant twice, once with a normal constraint
  //       and once with a reduction specialized constraint; 2) we always
  //       use reduction privilege and have the runtime promote it to
  //       read write privilege.
  static const int REGIONS = 0;

 public:
  static void cpu_variant(TaskContext &context);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(TaskContext &context);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(TaskContext &context);
#endif
};

}  // namespace numpy
}  // namespace legate
