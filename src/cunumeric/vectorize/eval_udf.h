/* Copyright 2023 NVIDIA Corporation
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
#include "core/data/scalar.h"

namespace cunumeric {

struct EvalUdfArgs {
  uint64_t cpu_func_ptr;
  std::vector<Array>& inputs;
  std::vector<Array>& outputs;
  std::vector<legate::Scalar>scalars;
  std::string ptx = "";
  uint32_t num_outputs;
  int64_t hash=0;
};

class EvalUdfTask : public CuNumericTask<EvalUdfTask> {
 public:
  static const int TASK_ID = CUNUMERIC_EVAL_UDF;

 public:
  static void cpu_variant(legate::TaskContext& context);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& context);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

}  // namespace cunumeric
