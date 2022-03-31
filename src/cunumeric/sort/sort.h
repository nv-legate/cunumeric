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

struct SortArgs {
  const Array& input;
  Array& output;
  bool argsort;
  bool stable;
  size_t sort_dim_size;
  bool is_index_space;
  size_t local_rank;
  size_t num_ranks;
};

template <typename VAL>
struct SampleEntry {
  VAL value;
  size_t rank;
  size_t local_id;
};

class SortTask : public CuNumericTask<SortTask> {
 public:
  static const int TASK_ID = CUNUMERIC_SORT;

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
