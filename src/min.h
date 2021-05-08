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

#ifndef __NUMPY_MIN_H__
#define __NUMPY_MIN_H__

#include "numpy.h"

#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

namespace legate {
namespace numpy {

// For doing min into particular dimensions
template <typename T>
class MinTask : public NumPyTask<MinTask<T>> {
 public:
  static const int TASK_ID;
  static const int REGIONS = 2;
#if 0
    public:
      template<typename TASK>
      static void set_layout_constraints(LegateVariant variant, 
                  Legion::TaskLayoutConstraintSet &layout_constraints);
#endif
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

// For doing a reduction to a single value
template <typename T>
class MinReducTask : public NumPyTask<MinReducTask<T>> {
 public:
  static const int TASK_ID;
  static const int REGIONS = 1;

 public:
  static T cpu_variant(const Legion::Task* task,
                       const std::vector<Legion::PhysicalRegion>& regions,
                       Legion::Context ctx,
                       Legion::Runtime* runtime);
#ifdef LEGATE_USE_OPENMP
  static T omp_variant(const Legion::Task* task,
                       const std::vector<Legion::PhysicalRegion>& regions,
                       Legion::Context ctx,
                       Legion::Runtime* runtime);
#endif
#ifdef LEGATE_USE_CUDA
  static Legion::DeferredReduction<Legion::MinReduction<T>> gpu_variant(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* runtime);
#endif
};

template <typename T>
class MinRadixTask : public NumPyTask<MinRadixTask<T>> {
 public:
  static const int TASK_ID;
  static const int REGIONS = MAX_REDUCTION_RADIX;

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

template <typename T>
class MinScalarTask : public NumPyTask<MinScalarTask<T>> {
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

#endif  // __NUMPY_MIN_H__
