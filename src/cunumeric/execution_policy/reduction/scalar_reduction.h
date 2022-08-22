#ifndef SRC_CUNUMERIC_EXECUTION_POLICY_REDUCTION_REDUCTION_H_
#define SRC_CUNUMERIC_EXECUTION_POLICY_REDUCTION_REDUCTION_H_

#include "cunumeric/cunumeric.h"
#include "cunumeric/omp_help.h"

#ifdef LEGATE_USE_OPENMP
#include <omp.h>
#endif


namespace cunumeric {
namespace scalar_reduction_impl {

template <VariantKind KIND>
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

template <>
struct ScalarReductionPolicy<VariantKind::CPU> {
  template <class AccessorRD, class LHS, class Kernel>
  void operator()(size_t volume, AccessorRD& out, const LHS& identity, Kernel&& kernel)
  {
    auto result = identity;
    for (size_t idx = 0; idx < volume; ++idx) { kernel(result, idx); }
    out.reduce(0, result);
  }
};

#ifdef LEGATE_USE_OPENMP
template <>
struct ScalarReductionPolicy<VariantKind::OMP> {
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
      for (size_t idx = 0; idx < volume; ++idx) { kernel(locals[tid], idx); }
    }
    for (auto idx = 0; idx < max_threads; ++idx) out.reduce(0, locals[idx]);
  }
};
#endif // LEGATE_USE_OPENMP

#ifdef LEGATE_USE_CUDA

template <class AccessorRD, class Kernel, class LHS>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  scalar_unary_red_kernel(size_t volume, size_t iters, AccessorRD out, Kernel kernel, LHS identity)
{
  auto value = identity;
  for (size_t idx = 0; idx < iters; idx++) {
    const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < volume) { kernel(value, offset); }
  }
  // Every thread in the thread block must participate in the exchange to get correct results
  reduce_output(out, value);
}

template <typename Buffer, typename RedAcc>
static __global__ void __launch_bounds__(1, 1) copy_kernel(Buffer result, RedAcc out)
{
  out.reduce(0, result.read());
}



template <>
struct ScalarReductionPolicy<VariantKind::GPU> {
  template <class AccessorRD, class LHS, class Kernel>
  void operator()(size_t volume, AccessorRD& out, const LHS& identity, Kernel&& kernel)
  {
    auto stream = get_cached_stream();

    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    DeferredReduction<typename OP::OP> result;
    size_t shmem_size = THREADS_PER_BLOCK / 32 * sizeof(LHS);

    if (blocks >= MAX_REDUCTION_CTAS) {
      const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
      scalar_unary_red_kernel<<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, iters, result, std::forward<Kernel>(kernel), identity);
    } else {
      scalar_unary_red_kernel<<<blocks, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, iters, result, std::forward<Kernel>(kernel), identity);
    }
    copy_kernel<<<1, 1, 0, stream>>>(result, out);
    CHECK_CUDA_STREAM(stream);
  }
};
#endif

}  // namespace scalar_reduction_impl
}  // namespace cunumeric

#endif  // SRC_CUNUMERIC_EXECUTION_POLICY_REDUCTION_REDUCTION_H_