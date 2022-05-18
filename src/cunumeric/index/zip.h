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

struct ZipArgs {
  const Array& out;
  const std::vector<Array>& inputs;
  const int64_t N;
  const int64_t key_dim;
  const int64_t start_index;
  const Legion::DomainPoint shape;
};

class ZipTask : public CuNumericTask<ZipTask> {
 public:
  static const int TASK_ID = CUNUMERIC_ZIP;

 public:
  static void cpu_variant(legate::TaskContext& context);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& context);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

constexpr coord_t compute_idx(coord_t index, coord_t dim)
{
  return index < 0 ? index + dim : index;
}

}  // namespace cunumeric
