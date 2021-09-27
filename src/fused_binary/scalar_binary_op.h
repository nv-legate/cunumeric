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
#include "scalar.h"

namespace legate {
namespace numpy {

class ScalarBinaryOpTask : public NumPyTask<ScalarBinaryOpTask> {
 public:
  static const int TASK_ID = NUMPY_SCALAR_BINARY_OP;

 public:
  static UntypedScalar cpu_variant(TaskContext& contex);
};

}  // namespace numpy
}  // namespace legate
