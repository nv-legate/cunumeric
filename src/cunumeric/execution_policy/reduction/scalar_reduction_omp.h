#ifndef SRC_CUNUMERIC_EXECUTION_POLICY_REDUCTION_SCALAR_REDUCTION_OMP_H_
#define SRC_CUNUMERIC_EXECUTION_POLICY_REDUCTION_SCALAR_REDUCTION_OMP_H_

#include "cunumeric/execution_policy/reduction/scalar_reduction.h"
#include "cunumeric/omp_help.h"

#include <omp.h>

namespace cunumeric {

template <class LG_OP, class Tag>
struct ScalarReductionPolicy<VariantKind::OMP, LG_OP, Tag> {
  template <class AccessorRD, class LHS, class Kernel>
  void operator()(size_t volume, AccessorRD& out, const LHS& identity, Kernel&& kernel)
  {
    const auto max_threads = omp_get_max_threads();
    ThreadLocalStorage<LHS> locals(max_threads);
    for (auto idx = 0; idx < max_threads; ++idx) locals[idx] = identity;
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) { kernel(locals[tid], idx, Tag{}); }
    }
    for (auto idx = 0; idx < max_threads; ++idx) out.reduce(0, locals[idx]);
  }
};

}  // namespace cunumeric

#endif  // SRC_CUNUMERIC_EXECUTION_POLICY_REDUCTION_SCALAR_REDUCTION_OMP_H_