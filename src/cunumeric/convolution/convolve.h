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

#pragma once

#include "cunumeric/cunumeric.h"

// We'll make some assumptions here about cache size
// that should hold up against most CPUs out there today
// Most L1 caches are 32-48KB so we'll go with the lower bound
#define L1_CACHE_SIZE 32768
// Most L2 caches are at least 256KB
#define L2_CACHE_SIZE 262144
// Most caches have 64B lines
#define CACHE_LINE_SIZE 64

namespace cunumeric {

struct ConvolveArgs {
  Array out;
  Array filter;
  std::vector<Array> inputs;
  legate::Domain root_domain;
};

class ConvolveTask : public CuNumericTask<ConvolveTask> {
 public:
  static const int TASK_ID = CUNUMERIC_CONVOLVE;

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
