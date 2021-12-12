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

#include "cunumeric/stat/bincount.h"
#include "cunumeric/stat/bincount_template.inl"

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;

template <typename VAL>
static __device__ inline void _bincount(int32_t* bins,
                                        AccessorRO<VAL, 1> rhs,
                                        const size_t volume,
                                        const size_t num_bins,
                                        Point<1> origin)
{
  // Initialize the bins to 0
  for (int32_t bin = threadIdx.x; bin < num_bins; bin += blockDim.x) bins[bin] = 0;
  __syncthreads();

  // Start reading values and do atomic updates to shared
  // Since these are 32 bit counts then we know they are native
  // atomics and willl therefore be "fast"
  size_t offset       = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;
  while (offset < volume) {
    const auto x   = origin[0] + offset;
    const auto bin = rhs[x];
    assert(bin < num_bins);
    SumReduction<int32_t>::fold<false>(bins[bin], 1);
    // Now get the next offset
    offset += stride;
  }
  // Wait for everyone to be done
  __syncthreads();
}

template <typename VAL>
static __device__ inline void _weighted_bincount(double* bins,
                                                 AccessorRO<VAL, 1> rhs,
                                                 AccessorRO<double, 1> weights,
                                                 const size_t volume,
                                                 const size_t num_bins,
                                                 Point<1> origin)
{
  // Initialize the bins to 0
  for (int32_t bin = threadIdx.x; bin < num_bins; bin += blockDim.x) bins[bin] = 0;
  __syncthreads();

  // Start reading values and do atomic updates to shared
  // Since these are 32 bit counts then we know they are native
  // atomics and willl therefore be "fast"
  size_t offset       = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;
  while (offset < volume) {
    const auto x   = origin[0] + offset;
    const auto bin = rhs[x];
    assert(bin < num_bins);
    SumReduction<double>::fold<false>(bins[bin], weights[x]);
    // Now get the next offset
    offset += stride;
  }
  // Wait for everyone to be done
  __syncthreads();
}

template <typename VAL>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  bincount_kernel_rd(AccessorRD<SumReduction<int64_t>, false, 1> lhs,
                     AccessorRO<VAL, 1> rhs,
                     const size_t volume,
                     const size_t num_bins,
                     Point<1> origin)
{
  extern __shared__ char array[];
  auto bins = reinterpret_cast<int32_t*>(array);
  _bincount(bins, rhs, volume, num_bins, origin);
  // Now do the atomics out to global memory
  for (int32_t bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    const auto count = bins[bin];
    if (count > 0) lhs.reduce(bin, count);
  }
}

template <typename VAL>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  bincount_kernel_rw(AccessorRW<int64_t, 1> lhs,
                     AccessorRO<VAL, 1> rhs,
                     const size_t volume,
                     const size_t num_bins,
                     Point<1> origin)
{
  extern __shared__ char array[];
  auto bins = reinterpret_cast<int32_t*>(array);
  _bincount(bins, rhs, volume, num_bins, origin);
  // Now do the atomics out to global memory
  for (int32_t bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    const auto count = bins[bin];
    if (count > 0) SumReduction<int64_t>::fold<false>(lhs[bin], count);
  }
}

template <typename VAL>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  weighted_bincount_kernel_rd(AccessorRD<SumReduction<double>, false, 1> lhs,
                              AccessorRO<VAL, 1> rhs,
                              AccessorRO<double, 1> weights,
                              const size_t volume,
                              const size_t num_bins,
                              Point<1> origin)
{
  extern __shared__ char array[];
  auto bins = reinterpret_cast<double*>(array);
  _weighted_bincount(bins, rhs, weights, volume, num_bins, origin);
  // Now do the atomics out to global memory
  for (int32_t bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    const auto weight = bins[bin];
    lhs.reduce(bin, weight);
  }
}

template <typename VAL>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  weighted_bincount_kernel_rw(AccessorRW<double, 1> lhs,
                              AccessorRO<VAL, 1> rhs,
                              AccessorRO<double, 1> weights,
                              const size_t volume,
                              const size_t num_bins,
                              Point<1> origin)
{
  extern __shared__ char array[];
  auto bins = reinterpret_cast<double*>(array);
  _weighted_bincount(bins, rhs, weights, volume, num_bins, origin);
  // Now do the atomics out to global memory
  for (int32_t bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    const auto weight = bins[bin];
    SumReduction<double>::fold<false>(lhs[bin], weight);
  }
}

template <LegateTypeCode CODE>
struct BincountImplBody<VariantKind::GPU, CODE> {
  using VAL = legate_type_of<CODE>;

  void operator()(AccessorRD<SumReduction<int64_t>, false, 1> lhs,
                  const AccessorRO<VAL, 1>& rhs,
                  const Rect<1>& rect,
                  const Rect<1>& lhs_rect) const
  {
    const auto volume   = rect.volume();
    const auto num_bins = lhs_rect.volume();
    const auto bin_size = num_bins * sizeof(int32_t);

    int32_t num_ctas = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_ctas, bincount_kernel_rd<VAL>, THREADS_PER_BLOCK, bin_size);
    assert(num_ctas > 0);
    // Launch a kernel with this number of CTAs
    bincount_kernel_rd<VAL>
      <<<num_ctas, THREADS_PER_BLOCK, bin_size>>>(lhs, rhs, volume, num_bins, rect.lo);
  }

  void operator()(const AccessorRW<int64_t, 1>& lhs,
                  const AccessorRO<VAL, 1>& rhs,
                  const Rect<1>& rect,
                  const Rect<1>& lhs_rect) const
  {
    const auto volume   = rect.volume();
    const auto num_bins = lhs_rect.volume();
    const auto bin_size = num_bins * sizeof(int32_t);

    int32_t num_ctas = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_ctas, bincount_kernel_rw<VAL>, THREADS_PER_BLOCK, bin_size);
    assert(num_ctas > 0);
    // Launch a kernel with this number of CTAs
    bincount_kernel_rw<VAL>
      <<<num_ctas, THREADS_PER_BLOCK, bin_size>>>(lhs, rhs, volume, num_bins, rect.lo);
  }

  void operator()(AccessorRD<SumReduction<double>, false, 1> lhs,
                  const AccessorRO<VAL, 1>& rhs,
                  const AccessorRO<double, 1>& weights,
                  const Rect<1>& rect,
                  const Rect<1>& lhs_rect) const
  {
    const auto volume   = rect.volume();
    const auto num_bins = lhs_rect.volume();
    const auto bin_size = num_bins * sizeof(double);

    int32_t num_ctas = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_ctas, weighted_bincount_kernel_rd<VAL>, THREADS_PER_BLOCK, bin_size);
    assert(num_ctas > 0);
    // Launch a kernel with this number of CTAs
    weighted_bincount_kernel_rd<VAL>
      <<<num_ctas, THREADS_PER_BLOCK, bin_size>>>(lhs, rhs, weights, volume, num_bins, rect.lo);
  }

  void operator()(const AccessorRW<double, 1>& lhs,
                  const AccessorRO<VAL, 1>& rhs,
                  const AccessorRO<double, 1>& weights,
                  const Rect<1>& rect,
                  const Rect<1>& lhs_rect) const
  {
    const auto volume   = rect.volume();
    const auto num_bins = lhs_rect.volume();
    const auto bin_size = num_bins * sizeof(double);

    int32_t num_ctas = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_ctas, weighted_bincount_kernel_rw<VAL>, THREADS_PER_BLOCK, bin_size);
    assert(num_ctas > 0);
    // Launch a kernel with this number of CTAs
    weighted_bincount_kernel_rw<VAL>
      <<<num_ctas, THREADS_PER_BLOCK, bin_size>>>(lhs, rhs, weights, volume, num_bins, rect.lo);
  }
};

/*static*/ void BincountTask::gpu_variant(TaskContext& context)
{
  bincount_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
