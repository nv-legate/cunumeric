#ifndef SRC_CUNUMERIC_EXECUTION_POLICY_REDUCTION_REDUCTION_H_
#define SRC_CUNUMERIC_EXECUTION_POLICY_REDUCTION_REDUCTION_H_

#include "cunumeric/cunumeric.h"

namespace cunumeric {
namespace scalar_reduction_impl {

template <VariantKind KIND, class LG_OP>
struct ScalarReductionPolicy {
  // No C++-20 yet. This is just here to illustrate the expected concept
  // that all kernels passed to this execution should have.
  struct KernelConcept {
    // Every operator should take a scalar LHS as the
    // target of the reduction and an index represeting the point
    // in the iteration space being added into the reduction.
    template <class LHS>
    void operator()(LHS& lhs, size_t idx)
    {
      // LHS <- op[idx]
    }
  };
};

template <class LG_OP>
struct ScalarReductionPolicy<VariantKind::CPU, LG_OP> {
  template <class AccessorRD, class LHS, class Kernel>
  void operator()(size_t volume, AccessorRD& out, const LHS& identity, Kernel&& kernel)
  {
    auto result = identity;
    for (size_t idx = 0; idx < volume; ++idx) { kernel(result, idx); }
    out.reduce(0, result);
  }
};

}  // namespace scalar_reduction_impl
}  // namespace cunumeric

#endif  // SRC_CUNUMERIC_EXECUTION_POLICY_REDUCTION_REDUCTION_H_