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
  const legate::DomainPoint shape;  // shape of the original array
  const bool has_input;
  const bool check_bounds;
  const Array& in = Array();
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

__CUDA_HD__ static int64_t compute_idx(const int64_t i, const int64_t volume, const bool&)
{
  return i % volume;
}

__CUDA_HD__ static int64_t compute_idx(const int64_t i,
                                       const int64_t volume,
                                       const legate::AccessorRO<int64_t, 1>& indices)
{
  int64_t idx   = indices[i];
  int64_t index = idx < 0 ? idx + volume : idx;
  return index;
}

static void check_idx(const int64_t i,
                      const int64_t volume,
                      const legate::AccessorRO<int64_t, 1>& indices)
{
  int64_t idx   = indices[i];
  int64_t index = idx < 0 ? idx + volume : idx;
  if (index < 0 || index >= volume)
    throw legate::TaskException("index is out of bounds in index array");
}
static void check_idx(const int64_t i, const int64_t volume, const bool&)
{
  // don't do anything when wrapping indices
}

static bool check_idx_omp(const int64_t i,
                          const int64_t volume,
                          const legate::AccessorRO<int64_t, 1>& indices)
{
  int64_t idx   = indices[i];
  int64_t index = idx < 0 ? idx + volume : idx;
  return (index < 0 || index >= volume);
}
static bool check_idx_omp(const int64_t i, const int64_t volume, const bool&) { return false; }

}  // namespace cunumeric
