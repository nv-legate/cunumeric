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

struct WrapArgs {
  const Array& out;                 // Array with Point<N> type that is used to
                                    // copy information from original array to the
                                    //  `wrapped` one
  const Legion::DomainPoint shape;  // shape of the original array
};

class WrapTask : public CuNumericTask<WrapTask> {
 public:
  static const int TASK_ID = CUNUMERIC_WRAP;

 public:
  static void cpu_variant(legate::TaskContext& context);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& context);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

constexpr coord_t compute_idx(coord_t index, coord_t extent)
{
  coord_t new_index = index < 0 ? index + extent : index;
  if (new_index < 0 || new_index >= extent)
    throw legate::TaskException("index is out of bounds in index array");
  return new_index;
}

}  // namespace cunumeric
