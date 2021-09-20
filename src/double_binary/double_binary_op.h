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
#include "double_binary/double_binary_op_util.h"

namespace legate {
namespace numpy {

struct DoubleBinaryOpArgs {
  const Array& in1;
  const Array& in2;
  const Array& in3;
  const Array& temp;
  const Array& out;
  DoubleBinaryOpCode op_code;
  std::vector<UntypedScalar> args;
};


struct DoubleBinaryOpArgs2 {
  std::vector<Array> ins;
  std::vector<Array> outs;
  std::vector<int> inStarts;
  std::vector<int> inSizes;
  std::vector<int> outStarts;
  std::vector<int> outSizes;
  std::vector<DoubleBinaryOpCode> op_codes;
  std::vector<UntypedScalar> args;
};



class DoubleBinaryOpTask : public NumPyTask<DoubleBinaryOpTask> {
 public:
  static const int TASK_ID = NUMPY_DOUBLE_BINARY_OP;

 public:
  static void cpu_variant(TaskContext& context);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(TaskContext& context);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(TaskContext& context);
#endif
};

}  // namespace numpy
}  // namespace legate
