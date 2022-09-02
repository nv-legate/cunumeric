#ifndef SRC_CUNUMERIC_EXECUTION_POLICY_REDUCTION_SCALAR_REDUCTION_CUH_
#define SRC_CUNUMERIC_EXECUTION_POLICY_REDUCTION_SCALAR_REDUCTION_CUH_

#include "cunumeric/execution_policy/reduction/scalar_reduction.h"
#include "cunumeric/cuda_help.h"

namespace cunumeric {
namespace scalar_reduction_impl {

template <class AccessorRD, class Kernel, class LHS, class Tag>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  scalar_unary_red_kernel(size_t volume, size_t iters, AccessorRD out, Kernel kernel, LHS identity, Tag tag)
{
  auto value = identity;
  for (size_t idx = 0; idx < iters; idx++) {
    const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < volume) { kernel(value, offset, tag); }
  }
  // Every thread in the thread block must participate in the exchange to get correct results
  reduce_output(out, value);
}


template <typename Buffer, typename RedAcc>
static __global__ void __launch_bounds__(1, 1) copy_kernel(Buffer result, RedAcc out)
{
  out.reduce(0, result.read());
}

} // namespace scalar_reduction_impl

template <class LG_OP, class Tag>
struct ScalarReductionPolicy<VariantKind::GPU, LG_OP, Tag> {
  template <class AccessorRD, class LHS, class Kernel>
  void __attribute__((visibility("hidden"))) operator()(size_t volume, AccessorRD& out, const LHS& identity, Kernel&& kernel)
  {
    auto stream = get_cached_stream();

    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    DeviceScalarReductionBuffer<LG_OP> result(stream);
    size_t shmem_size = THREADS_PER_BLOCK / 32 * sizeof(LHS);

    if (blocks >= MAX_REDUCTION_CTAS) {
      const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
      scalar_reduction_impl::scalar_unary_red_kernel<<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, iters, result, std::forward<Kernel>(kernel), identity, Tag{});
    } else {
      scalar_reduction_impl::scalar_unary_red_kernel<<<blocks, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, 1, result, std::forward<Kernel>(kernel), identity, Tag{});
    }
    scalar_reduction_impl::copy_kernel<<<1, 1, 0, stream>>>(result, out);
    CHECK_CUDA_STREAM(stream);
  }
};

} // namespace cunumeric

#endif // SRC_CUNUMERIC_EXECUTION_POLICY_REDUCTION_SCALAR_REDUCTION_CUH_