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

#include "cunumeric/cunumeric.h"
#include "cunumeric/unary/scalar_unary_red.h"
#include "cunumeric/unary/scalar_unary_red_template.inl"
#include "cunumeric/execution_policy/reduction/scalar_reduction.cuh"

namespace cunumeric {

using namespace legate;

/*static*/ void ScalarUnaryRedTask::gpu_variant(TaskContext& context)
{
  scalar_unary_red_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
