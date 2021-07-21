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

#pragma once

#include "legate.h"
#include "legate_numpy_c.h"
#include "scalar.h"
#include "mathtypes/complex.h"
#include <deque>

namespace legate {
namespace numpy {

using Array = Store;

enum class VariantKind : int {
  CPU = 0,
  OMP = 1,
  GPU = 2,
};

class LegateNumPy {
 public:
  // Record variants for all our tasks
  static void record_variant(Legion::TaskID tid,
                             const char* task_name,
                             const Legion::CodeDescriptor& desc,
                             Legion::ExecutionConstraintSet& execution_constraints,
                             Legion::TaskLayoutConstraintSet& layout_constraints,
                             LegateVariantCode var,
                             Legion::Processor::Kind kind,
                             bool leaf,
                             bool inner,
                             bool idempotent,
                             size_t ret_size);

 public:
  struct PendingTaskVariant : public Legion::TaskVariantRegistrar {
   public:
    PendingTaskVariant(void)
      : Legion::TaskVariantRegistrar(), task_name(NULL), var(LEGATE_NO_VARIANT)
    {
    }
    PendingTaskVariant(Legion::TaskID tid,
                       bool global,
                       const char* var_name,
                       const char* t_name,
                       const Legion::CodeDescriptor& desc,
                       LegateVariantCode v,
                       size_t ret)
      : Legion::TaskVariantRegistrar(tid, global, var_name),
        task_name(t_name),
        descriptor(desc),
        var(v),
        ret_size(ret)
    {
    }

   public:
    const char* task_name;
    Legion::CodeDescriptor descriptor;
    LegateVariantCode var;
    size_t ret_size;
  };
  static std::deque<PendingTaskVariant>& get_pending_task_variants(void);
};

template <typename T>
class NumPyTask : public LegateTask<T> {
 public:
  // Record variants for all our tasks
  static void record_variant(Legion::TaskID tid,
                             const Legion::CodeDescriptor& desc,
                             Legion::ExecutionConstraintSet& execution_constraints,
                             Legion::TaskLayoutConstraintSet& layout_constraints,
                             LegateVariantCode var,
                             Legion::Processor::Kind kind,
                             bool leaf,
                             bool inner,
                             bool idempotent,
                             size_t ret_size)
  {
    // For this just turn around and call this on the base LegateNumPy
    // type so it will deduplicate across all task kinds
    LegateNumPy::record_variant(tid,
                                NumPyTask<T>::task_name(),
                                desc,
                                execution_constraints,
                                layout_constraints,
                                var,
                                kind,
                                leaf,
                                inner,
                                idempotent,
                                ret_size);
  }
};

}  // namespace numpy
}  // namespace legate
