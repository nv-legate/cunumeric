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

#ifndef __NUMPY_MOD_H__
#define __NUMPY_MOD_H__

#include "numpy.h"

namespace legate {
namespace numpy {

// Standard data-parallel modulus task for integers
template <typename T>
class IntModTask : public NumPyTask<IntModTask<T>> {
 public:
  static const int TASK_ID;
  static const int REGIONS = 3;

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

// Standard data-parallel modulus task for floats
template <typename T>
class RealModTask : public NumPyTask<RealModTask<T>> {
 public:
  static const int TASK_ID;
  static const int REGIONS = 3;

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

// Mod task for handling broadcasting of dimensions for integers
template <typename T>
class IntModBroadcast : public NumPyTask<IntModBroadcast<T>> {
 public:
  static const int TASK_ID;
  static const int REGIONS = 3;

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

// Mod task for handling broadcasting of dimensions for floats
template <typename T>
class RealModBroadcast : public NumPyTask<RealModBroadcast<T>> {
 public:
  static const int TASK_ID;
  static const int REGIONS = 3;

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

// For doing a scalar modulus for integers
template <typename T>
class IntModScalar : public NumPyTask<IntModScalar<T>> {
 public:
  static const int TASK_ID;
  static const int REGIONS = 0;

 public:
  static T cpu_variant(const Legion::Task* task,
                       const std::vector<Legion::PhysicalRegion>& regions,
                       Legion::Context ctx,
                       Legion::Runtime* runtime);
};

// For doing a scalar modulus for floats
template <typename T>
class RealModScalar : public NumPyTask<RealModScalar<T>> {
 public:
  static const int TASK_ID;
  static const int REGIONS = 0;

 public:
  static T cpu_variant(const Legion::Task* task,
                       const std::vector<Legion::PhysicalRegion>& regions,
                       Legion::Context ctx,
                       Legion::Runtime* runtime);
};

}  // namespace numpy
}  // namespace legate

#endif  // __NUMPY_MOD_H__
