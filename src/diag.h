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

#ifndef __NUMPY_DIAG_H__
#define __NUMPY_DIAG_H__

#include "numpy.h"
#include "proj.h"

namespace legate {
namespace numpy {
// Small helper method for diagonal
static inline Legion::Rect<1> extract_rect1d(const Legion::Domain& dom)
{
  const Legion::Rect<2> rect = dom;
  Legion::Rect<1> result;
  result.lo[0] = rect.lo[0];
  result.hi[0] = rect.hi[0];
  return result;
}

template <typename T>
class DiagTask : public NumPyTask<DiagTask<T>> {
 public:
  static const int TASK_ID;
  static const int REGIONS = 2;

 public:
  template <typename TASK>
  static void set_layout_constraints(LegateVariant variant,
                                     Legion::TaskLayoutConstraintSet& layout_constraints);

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
}  // namespace numpy
}  // namespace legate

#endif  // __NUMPY_DIAG_H__
