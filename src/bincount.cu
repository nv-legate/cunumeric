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

#include "bincount.h"
#include "cuda_help.h"
#include "proj.h"

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
__global__ void legate_bincount_1d(const AccessorWO<uint64_t, 1> out, const AccessorRO<T, 1> in, const Point<1> origin,
                                   const size_t max, const T num_bins) {
  extern __shared__ char array[];
  int*                   bins = (int*)array;
  // Initialize the bins to 0
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x)
    bins[bin] = 0;
  __syncthreads();
  // Start reading values and do atomic updates to shared
  // Since these are 32 bit counts then we know they are native
  // atomics and willl therefore be "fast"
  size_t       offset = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;
  while (offset < max) {
    const coord_t x   = origin[0] + offset;
    const T       bin = in[x];
    assert(bin < num_bins);
    atomicAdd(bins + bin, 1);
    // Now get the next offset
    offset += stride;
  }
  // Wait for everyone to be done
  __syncthreads();
  // Now do the atomics out to global memory
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    const uint64_t count = bins[bin];
    if (count > 0) SumReduction<uint64_t>::fold<false /*exclusive*/>(out[bin], count);
  }
}

template<typename T>
__global__ void legate_bincount_2d(const AccessorWO<uint64_t, 1> out, const AccessorRO<T, 2> in, const Point<2> origin,
                                   const Point<1> pitch, const size_t max, const T num_bins) {
  extern __shared__ char array[];
  int*                   bins = (int*)array;
  // Initialize the bins to 0
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x)
    bins[bin] = 0;
  __syncthreads();
  // Start reading values and do atomic updates to shared
  // Since these are 32 bit counts then we know they are native
  // atomics and willl therefore be "fast"
  size_t       offset = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;
  while (offset < max) {
    const coord_t x   = origin[0] + offset / pitch[0];
    const coord_t y   = origin[1] + offset % pitch[0];
    const T       bin = in[x][y];
    assert(bin < num_bins);
    atomicAdd(bins + bin, 1);
    // Now get the next offset
    offset += stride;
  }
  // Wait for everyone to be done
  __syncthreads();
  // Now do the atomics out to global memory
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    const uint64_t count = bins[bin];
    if (count > 0) SumReduction<uint64_t>::fold<false /*exclusive*/>(out[bin], count);
  }
}

template<typename T>
__global__ void legate_bincount_3d(const AccessorWO<uint64_t, 1> out, const AccessorRO<T, 3> in, const Point<3> origin,
                                   const Point<2> pitch, const size_t max, const T num_bins) {
  extern __shared__ char array[];
  int*                   bins = (int*)array;
  // Initialize the bins to 0
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x)
    bins[bin] = 0;
  __syncthreads();
  // Start reading values and do atomic updates to shared
  // Since these are 32 bit counts then we know they are native
  // atomics and willl therefore be "fast"
  size_t       offset = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;
  while (offset < max) {
    const coord_t x   = origin[0] + offset / pitch[0];
    const coord_t y   = origin[1] + (offset % pitch[0]) / pitch[1];
    const coord_t z   = origin[2] + (offset % pitch[0]) % pitch[1];
    const T       bin = in[x][y][z];
    assert(bin < num_bins);
    atomicAdd(bins + bin, 1);
    // Now get the next offset
    offset += stride;
  }
  // Wait for everyone to be done
  __syncthreads();
  // Now do the atomics out to global memory
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    const uint64_t count = bins[bin];
    if (count > 0) SumReduction<uint64_t>::fold<false /*exclusive*/>(out[bin], count);
  }
}

template<typename T>
/*static*/ void BinCountTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                             Runtime* runtime) {
  LegateDeserializer            derez(task->args, task->arglen);
  const int                     collapse_dim   = derez.unpack_dimension();
  const int                     collapse_index = derez.unpack_dimension();
  const Rect<1>                 bin_rect       = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  const AccessorWO<uint64_t, 1> out =
      (collapse_dim >= 0)
          ? derez.unpack_accessor_WO<uint64_t, 1>(regions[0], bin_rect, collapse_dim, task->index_point[collapse_index])
          : derez.unpack_accessor_WO<uint64_t, 1>(regions[0], bin_rect);
  const size_t bin_volume = bin_rect.volume();
  // Initialize all the counts to zero
  cudaMemset(out.ptr(bin_rect), 0, bin_volume * sizeof(uint64_t));
  // Use 32-bit ints for bin counts in threadblocks for native atomics
  const size_t bin_size = bin_volume * sizeof(int32_t);
  const int    dim      = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
      // Figure out how many active blocks we can launch given the number
      // of bins and our normal number of threads per thread block
      int num_ctas = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_ctas, legate_bincount_1d<T>, THREADS_PER_BLOCK, bin_size);
      assert(num_ctas > 0);
      // Launch a kernel with this number of CTAs
      const size_t volume = rect.volume();
      legate_bincount_1d<T><<<num_ctas, THREADS_PER_BLOCK, bin_size>>>(out, in, rect.lo, volume, bin_volume);
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      // Figure out how many active blocks we can launch given the number
      // of bins and our normal number of threads per thread block
      int num_ctas = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_ctas, legate_bincount_2d<T>, THREADS_PER_BLOCK, bin_size);
      assert(num_ctas > 0);
      // Launch a kernel with this number of CTAs
      const size_t  volume = rect.volume();
      const coord_t pitch  = rect.hi[1] - rect.lo[1] + 1;
      legate_bincount_2d<T><<<num_ctas, THREADS_PER_BLOCK, bin_size>>>(out, in, rect.lo, pitch, volume, bin_volume);
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      // Figure out how many active blocks we can launch given the number
      // of bins and our normal number of threads per thread block
      int num_ctas = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_ctas, legate_bincount_3d<T>, THREADS_PER_BLOCK, bin_size);
      assert(num_ctas > 0);
      // Launch a kernel with this number of CTAs
      const size_t  volume   = rect.volume();
      const coord_t diffy    = rect.hi[1] - rect.lo[1] + 1;
      const coord_t diffz    = rect.hi[2] - rect.lo[2] + 1;
      const coord_t pitch[2] = {diffy * diffz, diffz};
      legate_bincount_3d<T><<<num_ctas, THREADS_PER_BLOCK, bin_size>>>(out, in, rect.lo, Point<2>(pitch), volume, bin_volume);
      break;
    }
    default:
      assert(false);
  }
}

INSTANTIATE_INT_VARIANT(BinCountTask, gpu_variant)
INSTANTIATE_UINT_VARIANT(BinCountTask, gpu_variant)

template<typename T, typename WT>
__global__ void legate_weighted_bincount_1d(const AccessorWO<WT, 1> out, const AccessorRO<T, 1> in, const AccessorRO<WT, 1> weights,
                                            const Point<1> origin, const size_t max, const T num_bins, const WT identity) {
  extern __shared__ char array[];
  WT*                    bins = (WT*)array;
  // Initialize the bins to 0
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x)
    bins[bin] = identity;
  __syncthreads();
  // Start reading values and do atomic updates to shared
  size_t       offset = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;
  while (offset < max) {
    const coord_t x   = origin[0] + offset;
    const T       bin = in[x];
    assert(bin < num_bins);
    SumReduction<WT>::template fold<false /*exclusive*/>(bins[bin], weights[x]);
    // Now get the next offset
    offset += stride;
  }
  // Wait for everyone to be done
  __syncthreads();
  // Now do the atomics out to global memory
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    const WT weight = bins[bin];
    if (weight != identity) SumReduction<WT>::template fold<false /*exclusive*/>(out[bin], bins[bin]);
  }
}

template<typename T, typename WT>
__global__ void legate_weighted_bincount_2d(const AccessorWO<WT, 1> out, const AccessorRO<T, 2> in, const AccessorRO<WT, 2> weights,
                                            const Point<2> origin, const Point<1> pitch, const size_t max, const T num_bins,
                                            const WT identity) {
  extern __shared__ char array[];
  WT*                    bins = (WT*)array;
  // Initialize the bins to 0
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x)
    bins[bin] = identity;
  __syncthreads();
  // Start reading values and do atomic updates to shared
  size_t       offset = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;
  while (offset < max) {
    const coord_t x   = origin[0] + offset / pitch[0];
    const coord_t y   = origin[1] + offset % pitch[0];
    const T       bin = in[x][y];
    assert(bin < num_bins);
    SumReduction<WT>::template fold<false /*exclusive*/>(bins[bin], weights[x][y]);
    // Now get the next offset
    offset += stride;
  }
  // Wait for everyone to be done
  __syncthreads();
  // Now do the atomics out to global memory
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    const WT weight = bins[bin];
    if (weight != identity) SumReduction<WT>::template fold<false /*exclusive*/>(out[bin], bins[bin]);
  }
}

template<typename T, typename WT>
__global__ void legate_weighted_bincount_3d(const AccessorWO<WT, 1> out, const AccessorRO<T, 3> in, const AccessorRO<WT, 3> weights,
                                            const Point<3> origin, const Point<2> pitch, const size_t max, const T num_bins,
                                            const WT identity) {
  extern __shared__ char array[];
  WT*                    bins = (WT*)array;
  // Initialize the bins to 0
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x)
    bins[bin] = identity;
  __syncthreads();
  // Start reading values and do atomic updates to shared
  size_t       offset = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;
  while (offset < max) {
    const coord_t x   = origin[0] + offset / pitch[0];
    const coord_t y   = origin[1] + (offset % pitch[0]) / pitch[1];
    const coord_t z   = origin[2] + (offset % pitch[0]) % pitch[1];
    const T       bin = in[x][y][z];
    assert(bin < num_bins);
    SumReduction<WT>::template fold<false /*exclusive*/>(bins[bin], weights[x][y][z]);
    // Now get the next offset
    offset += stride;
  }
  // Wait for everyone to be done
  __syncthreads();
  // Now do the atomics out to global memory
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    const WT weight = bins[bin];
    if (weight != identity) SumReduction<WT>::template fold<false /*exclusive*/>(out[bin], bins[bin]);
  }
}

template<typename T, typename WT>
/*static*/ void WeightedBinCountTask<T, WT>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                         Runtime* runtime) {
  LegateDeserializer      derez(task->args, task->arglen);
  const int               collapse_dim   = derez.unpack_dimension();
  const int               collapse_index = derez.unpack_dimension();
  const Rect<1>           bin_rect       = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  const AccessorWO<WT, 1> out =
      (collapse_dim >= 0) ? derez.unpack_accessor_WO<WT, 1>(regions[0], bin_rect, collapse_dim, task->index_point[collapse_index])
                          : derez.unpack_accessor_WO<WT, 1>(regions[0], bin_rect);
  const size_t bin_volume = bin_rect.volume();
  // Initialize all the counts to zero
  cudaMemset(out.ptr(bin_rect), 0, bin_volume * sizeof(WT));
  // Use 32-bit ints for bin counts in threadblocks for native atomics
  const size_t bin_size = bin_volume * sizeof(WT);
  const int    dim      = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1>  in      = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
      const AccessorRO<WT, 1> weights = derez.unpack_accessor_RO<WT, 1>(regions[2], rect);
      // Figure out how many active blocks we can launch given the number
      // of bins and our normal number of threads per thread block
      int num_ctas = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_ctas, legate_bincount_1d<T>, THREADS_PER_BLOCK, bin_size);
      assert(num_ctas > 0);
      // Launch a kernel with this number of CTAs
      const size_t volume = rect.volume();
      legate_weighted_bincount_1d<T, WT>
          <<<num_ctas, THREADS_PER_BLOCK, bin_size>>>(out, in, weights, rect.lo, volume, bin_volume, SumReduction<WT>::identity);
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2>  in      = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      const AccessorRO<WT, 2> weights = derez.unpack_accessor_RO<WT, 2>(regions[2], rect);
      // Figure out how many active blocks we can launch given the number
      // of bins and our normal number of threads per thread block
      int num_ctas = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_ctas, legate_bincount_2d<T>, THREADS_PER_BLOCK, bin_size);
      assert(num_ctas > 0);
      // Launch a kernel with this number of CTAs
      const size_t  volume = rect.volume();
      const coord_t pitch  = rect.hi[1] - rect.lo[1] + 1;
      legate_weighted_bincount_2d<T, WT><<<num_ctas, THREADS_PER_BLOCK, bin_size>>>(out, in, weights, rect.lo, pitch, volume,
                                                                                    bin_volume, SumReduction<WT>::identity);
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3>  in      = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      const AccessorRO<WT, 3> weights = derez.unpack_accessor_RO<WT, 3>(regions[2], rect);
      // Figure out how many active blocks we can launch given the number
      // of bins and our normal number of threads per thread block
      int num_ctas = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_ctas, legate_bincount_3d<T>, THREADS_PER_BLOCK, bin_size);
      assert(num_ctas > 0);
      // Launch a kernel with this number of CTAs
      const size_t  volume   = rect.volume();
      const coord_t diffy    = rect.hi[1] - rect.lo[1] + 1;
      const coord_t diffz    = rect.hi[2] - rect.lo[2] + 1;
      const coord_t pitch[2] = {diffy * diffz, diffz};
      legate_weighted_bincount_3d<T, WT><<<num_ctas, THREADS_PER_BLOCK, bin_size>>>(out, in, weights, rect.lo, Point<2>(pitch),
                                                                                    volume, bin_volume, SumReduction<WT>::identity);
      break;
    }
    default:
      assert(false);
  }
}

#define INSTANTIATE_WEIGHTED_BINCOUNT_VARIANT(task, type, variant)                                                 \
  template void task<type, __half>::variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);   \
  template void task<type, float>::variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);    \
  template void task<type, double>::variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);   \
  template void task<type, int16_t>::variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);  \
  template void task<type, int32_t>::variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);  \
  template void task<type, int64_t>::variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);  \
  template void task<type, uint16_t>::variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void task<type, uint32_t>::variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void task<type, uint64_t>::variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
// No bools for now

INSTANTIATE_WEIGHTED_BINCOUNT_VARIANT(WeightedBinCountTask, int16_t, gpu_variant)
INSTANTIATE_WEIGHTED_BINCOUNT_VARIANT(WeightedBinCountTask, int32_t, gpu_variant)
INSTANTIATE_WEIGHTED_BINCOUNT_VARIANT(WeightedBinCountTask, int64_t, gpu_variant)
INSTANTIATE_WEIGHTED_BINCOUNT_VARIANT(WeightedBinCountTask, uint16_t, gpu_variant)
INSTANTIATE_WEIGHTED_BINCOUNT_VARIANT(WeightedBinCountTask, uint32_t, gpu_variant)
INSTANTIATE_WEIGHTED_BINCOUNT_VARIANT(WeightedBinCountTask, uint64_t, gpu_variant)

}    // namespace numpy
}    // namespace legate
