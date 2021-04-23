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

#ifndef __NUMPY_ARG_H__
#define __NUMPY_ARG_H__

#include "numpy.h"

namespace legate {
namespace numpy {

template <typename T>
class Argval {
 public:
  __CUDA_HD__
  Argval(T value);
  __CUDA_HD__
  Argval(int64_t arg, T value);

 public:
  template <typename REDOP, bool EXCLUSIVE>
  __CUDA_HD__ inline void apply(const Argval<T>& rhs);

 public:
  int64_t arg;
  T arg_value;
};

// Data parallel get arg
template <typename T>
class GetargTask : public NumPyTask<GetargTask<T>> {
 public:
  static const int TASK_ID;
  static const int REGIONS = 2;

 public:
  static void cpu_variant(const Legion::Task* task,
                          const std::vector<Legion::PhysicalRegion>& regions,
                          Legion::Context ctx,
                          Legion::Runtime* runtime);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(const Legion::Task* task,
                          const std::vector<Legion::PhysicalRegion>& regions,
                          Legion::Context ctx,
                          Legion::Runtime* runtime);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(const Legion::Task* task,
                          const std::vector<Legion::PhysicalRegion>& regions,
                          Legion::Context ctx,
                          Legion::Runtime* runtime);
#endif
};

// For doing a scalar get arg
template <typename T>
class GetargScalar : public NumPyTask<GetargScalar<T>> {
 public:
  static const int TASK_ID;
  static const int REGIONS = 0;

 public:
  static int64_t cpu_variant(const Legion::Task* task,
                             const std::vector<Legion::PhysicalRegion>& regions,
                             Legion::Context ctx,
                             Legion::Runtime* runtime);
};

}  // namespace numpy
}  // namespace legate

#include "arg.inl"

#endif  // __NUMPY_ARG_H__
