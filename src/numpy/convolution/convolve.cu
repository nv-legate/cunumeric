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

#include "numpy/divmod.h"
#include "numpy/cuda_help.h"
#include "numpy/convolution/convolve.h"
#include "numpy/convolution/convolve_template.inl"

namespace legate {
namespace numpy {

using namespace Legion;

// For optimizing bandwidth utilization for convolution we load data
// from the input into shared memory and leave the filter in global memory
// with the expectation that it can be cached in the L2 and likely even
// the L1 cache of each SM across all threads and threadblocks. We stream
// reads from the inputs and stores from the outputs with the appropriate
// cache qualifiers in order to avoid polluting the filter data in the caches.

// We have several different variants of the convolution kernel to try to 
// minimize how often we load data. We do this by computing a "logical" tiling
// of the space. We want our logical tiling to be at least as wide as
// the filter in all dimensions so that we can minimize the amount of redundant
// data movement that needs to be done to perform the computation. We also require
// that the last dimension be loading at least contiguous bytes so we can get 
// coalesced loads. We begin by computing the logical tiling and seeing how
// much shared memory it requires:
// Case 1: The tiling requires less that SMEM_PER_CTA so we can fit the entire
//         computation in a threadblock. This is the nicest case and the one
//         that should result in the best performance since we'll be able to
//         load the data into shared memory and then have the threads loop
//         over all the points and compute their convolutions
// Case 2: We couldn't fit the whole tile in shared memory, but we could fit
//         all the dimensions except one (but not the last one) into shared 
//         memory. In this case we'll load tiles of N-1 dimensions and walk
//         along the remaining dimension, with threads maintaining a pencil
//         of values for the partial convolution computations of points in
//         that particular dimension. 
// Case 3: We couldn't fit even a subset of the tile in shared memory. In
//         this case we will just punt and be at the mercy of the cache
//         gods in order to get any kind of locality by having each thread
//         just perform a convolution for an output point directly.

// For case 2 we use warp-specialization for loading the data so we can pipeline
// the loading of the data. We have two sub-variants of this kernel: one in which
// the tile can fit into SMEM_PER_CTA/2 shared memory so we can double buffer 
// for improved memory latency hiding and one in which it can't so we need to 
// single buffer the loading of data (but still preload into registers).

template<int DIM>
struct ConvolutionCase1Args {
  FastDivmodU64 grid_pitches[DIM];
  FastDivmodU64 block_pitches[DIM];
  FastDivmodU64 input_pitches[DIM];
  unsigned block_tiles[DIM];
  unsigned filter_centers[DIM];
  unsigned filter_extents[DIM];
  Point<DIM> delta_lo, delta_hi;
  size_t filter_volume;
  size_t tile_volume;
  size_t input_volume;
};

template<typename VAL, int DIM>
__global__ static void __launch_bounds__(512, 2)
convolution_case1a_kernel(const AccessorWO<VAL, DIM> out,
                          const AccessorRO<VAL, DIM> filter,
                          const AccessorRO<VAL, DIM> in,
                          const Rect<DIM> root_rect,
                          const Rect<DIM> subrect,
                          const Rect<DIM> filter_rect,
                          const ConvolutionCase1Args<DIM> args)
{
  // Deal with compiler shared memory stupidity
  extern __shared__ uint8_t buffer[];
  // Technically this illegal C++, but there's no other way to do it
  VAL *input = (VAL*)buffer;
  // Compute the origin point of the block
  size_t offset = blockIdx.x;
  Point<DIM> block_point = root_rect.lo;
  #pragma unroll
  for (int d = 0; d < DIM; d++)
    block_point[d] += args.grid_pitches[d].divmod(offset, offset) * args.block_tiles[d];
  // Load in the shared memory for this block
  Point<DIM> tile_point;
  const Rect<DIM> input_bounds(block_point - args.delta_lo, block_point + args.delta_hi);
  if (subrect.contains(input_bounds)) {
    // All the points are contained, so no need for point-wise tests
    // Unroll this four times to try to pipeline loads
    #pragma unroll 4
    for (unsigned idx = threadIdx.x; idx < args.input_volume; idx += blockDim.x) {
      offset = idx;
      #pragma unroll
      for (int d = 0; d < DIM; d++)
        tile_point[d] = args.input_pitches[d].divmod(offset,offset);
      VAL value = in[input_bounds.lo + tile_point];
      // Write the value into shared memory
      input[idx] = value;
    }
  } else {
    // Need to do point-wise tests
    // Unroll this four times to try to pipeline loads
    #pragma unroll 4
    for (unsigned idx = threadIdx.x; idx < args.input_volume; idx += blockDim.x) {
      offset = idx;
      #pragma unroll
      for (int d = 0; d < DIM; d++)
        tile_point[d] = args.input_pitches[d].divmod(offset,offset);
      if (!subrect.contains(input_bounds.lo + tile_point))
        continue;
      VAL value = in[input_bounds.lo + tile_point];
      // Write the value into shared memory
      input[idx] = value;
    }
  }
  // Wait for everything to be loaded into shared memory 
  __syncthreads();
  // Loop over points in the tile and compute the outputs
  coord_t f_coords[DIM];
  Point<DIM> out_point, in_point, filter_point;
  for (unsigned idx = threadIdx.x; idx < args.tile_volume; idx += blockDim.x) {
    // Compute the local coordinates
    offset = idx;
    #pragma unroll
    for (int d = 0; d < DIM; d++) {
      tile_point[d] = args.block_pitches[d].divmod(offset, offset); 
      out_point[d] = block_point[d] + tile_point[d];
    }
    if (!root_rect.contains(out_point))
      continue;
    #pragma unroll
    for (int d = 0; d < DIM; d++)
      f_coords[d] = 0;
    VAL acc{0};
    for (unsigned idx = 0; idx < args.filter_volume; idx++) {
      #pragma unroll
      for (int d = 0; d < DIM; d++)
        in_point[d] = out_point[d] + f_coords[d] - args.filter_centers[d];
      if (subrect.contains(in_point))
      {
        offset = 0;
        #pragma unroll
        for (int d = 0; d < DIM; d++)
          offset += (tile_point[d] + f_coords[d]) * args.input_pitches[d].divisor; 
        #pragma unroll
        for (int d = 0; d < DIM; d++)
          filter_point[d] = args.filter_extents[d] - f_coords[d] - 1;
        acc = acc + input[offset] * filter[filter_point];
      }
      // Step the filter coordinates
      #pragma unroll
      for (int d = DIM-1; d >= 0; d--) {
        f_coords[d]++;
        if (f_coords[d] == args.filter_extents[d])
          f_coords[d] = 0;
        else
          break;
      }
    }
    out[out_point] = acc;
  }
}

// This version of the kernel is identical to the one above but with
// different launch bounds to handle a bigger CTA with more shared memory
template<typename VAL, int DIM>
__global__ static void __launch_bounds__(1024, 1)
convolution_case1b_kernel(const AccessorWO<VAL, DIM> out,
                          const AccessorRO<VAL, DIM> filter,
                          const AccessorRO<VAL, DIM> in,
                          const Rect<DIM> root_rect,
                          const Rect<DIM> subrect,
                          const Rect<DIM> filter_rect,
                          const ConvolutionCase1Args<DIM> args)
{
  // Deal with compiler shared memory stupidity
  extern __shared__ uint8_t buffer[];
  // Technically this illegal C++, but there's no other way to do it
  VAL *input = (VAL*)buffer;
  // Compute the origin point of the block
  size_t offset = blockIdx.x;
  Point<DIM> block_point = root_rect.lo;
  #pragma unroll
  for (int d = 0; d < DIM; d++)
    block_point[d] += args.grid_pitches[d].divmod(offset, offset) * args.block_tiles[d];
  // Load in the shared memory for this block
  Point<DIM> tile_point;
  const Rect<DIM> input_bounds(block_point - args.delta_lo, block_point + args.delta_hi);
  if (subrect.contains(input_bounds)) {
    // All the points are contained, so no need for point-wise tests
    // Unroll this four times to try to pipeline loads
    #pragma unroll 4
    for (unsigned idx = threadIdx.x; idx < args.input_volume; idx += blockDim.x) {
      offset = idx;
      #pragma unroll
      for (int d = 0; d < DIM; d++)
        tile_point[d] = args.input_pitches[d].divmod(offset,offset);
      VAL value = in[input_bounds.lo + tile_point];
      // Write the value into shared memory
      input[idx] = value;
    }
  } else {
    // Need to do point-wise tests
    // Unroll this four times to try to pipeline loads
    #pragma unroll 4
    for (unsigned idx = threadIdx.x; idx < args.input_volume; idx += blockDim.x) {
      offset = idx;
      #pragma unroll
      for (int d = 0; d < DIM; d++)
        tile_point[d] = args.input_pitches[d].divmod(offset,offset);
      if (!subrect.contains(input_bounds.lo + tile_point))
        continue;
      VAL value = in[input_bounds.lo + tile_point];
      // Write the value into shared memory
      input[idx] = value;
    }
  }
  // Wait for everything to be loaded into shared memory 
  __syncthreads();
  // Loop over points in the tile and compute the outputs
  coord_t f_coords[DIM];
  Point<DIM> out_point, in_point, filter_point;
  for (unsigned idx = threadIdx.x; idx < args.tile_volume; idx += blockDim.x) {
    // Compute the local coordinates
    offset = idx;
    #pragma unroll
    for (int d = 0; d < DIM; d++) {
      tile_point[d] = args.block_pitches[d].divmod(offset, offset); 
      out_point[d] = block_point[d] + tile_point[d];
    }
    if (!root_rect.contains(out_point))
      continue;
    #pragma unroll
    for (int d = 0; d < DIM; d++)
      f_coords[d] = 0;
    VAL acc{0};
    for (unsigned idx = 0; idx < args.filter_volume; idx++) {
      #pragma unroll
      for (int d = 0; d < DIM; d++)
        in_point[d] = out_point[d] + f_coords[d] - args.filter_centers[d];
      if (subrect.contains(in_point))
      {
        offset = 0;
        #pragma unroll
        for (int d = 0; d < DIM; d++)
          offset += (tile_point[d] + f_coords[d]) * args.input_pitches[d].divisor; 
        #pragma unroll
        for (int d = 0; d < DIM; d++)
          filter_point[d] = args.filter_extents[d] - f_coords[d] - 1;
        acc = acc + input[offset] * filter[filter_point];
      }
      // Step the filter coordinates
      #pragma unroll
      for (int d = DIM-1; d >= 0; d--) {
        f_coords[d]++;
        if (f_coords[d] == args.filter_extents[d])
          f_coords[d] = 0;
        else
          break;
      }
    }
    out[out_point] = acc;
  }
}

template<int DIM>
struct ConvolutionCase3Args {
  FastDivmodU64 grid_pitches[DIM];
  FastDivmodU64 block_pitches[DIM];
  unsigned block_tiles[DIM];
  unsigned filter_centers[DIM];
  unsigned filter_extents[DIM];
  size_t filter_volume;
};

template<typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, 2)
convolution_case3_kernel(const AccessorWO<VAL, DIM> out,
                         const AccessorRO<VAL, DIM> filter,
                         const AccessorRO<VAL, DIM> in,
                         const Rect<DIM> root_rect,
                         const Rect<DIM> subrect,
                         const Rect<DIM> filter_rect,
                         const ConvolutionCase3Args<DIM> args)
{
  // Compute our local point from our block and thread IDs
  Point<DIM> out_point = root_rect.lo;
  size_t offset = blockIdx.x;
  #pragma unroll
  for (int d = 0; d < DIM; d++)
    out_point[d] += args.grid_pitches[d].divmod(offset, offset) * args.block_tiles[d];
  offset = threadIdx.x;
  #pragma unroll
  for (int d = 0; d < DIM; d++)
    out_point[d] += args.block_pitches[d].divmod(offset, offset);
  // If we're not computing an output there is nothing for us to do
  if (!root_rect.contains(out_point))
    return;
  coord_t f_coords[DIM];
  #pragma unroll
  for (int d = 0; d < DIM; d++)
    f_coords[d] = 0;
  VAL acc{0};
  Point<DIM> in_point, filter_point;
  for (unsigned idx = 0; idx < args.filter_volume; idx++) {
    #pragma unroll
    for (int d = 0; d < DIM; d++)
      in_point[d] = out_point[d] + f_coords[d] - args.filter_centers[d];
    if (subrect.contains(in_point))
    {
      #pragma unroll
      for (int d = 0; d < DIM; d++)
        filter_point[d] = args.filter_extents[d] - f_coords[d] - 1;
      acc = acc + in[in_point] * filter[filter_point];
    }
    // Step the filter coordinates
    #pragma unroll
    for (int d = DIM-1; d >= 0; d--) {
      f_coords[d]++;
      if (f_coords[d] == args.filter_extents[d])
        f_coords[d] = 0;
      else
        break;
    }
  }
  out[out_point] = acc;
}

template <LegateTypeCode CODE, int DIM>
struct ConvolveImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  __host__
  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, DIM> filter,
                  AccessorRO<VAL, DIM> in,
                  const Rect<DIM>& root_rect,
                  const Rect<DIM>& subrect,
                  const Rect<DIM>& filter_rect) const
  {
    // Get the maximum amount of shared memory per threadblock
    int device;
    CHECK_CUDA( cudaGetDevice(&device) );
    cudaDeviceProp properties;
    CHECK_CUDA( cudaGetDeviceProperties(&properties, device) );
    size_t max_smem_size = properties.sharedMemPerBlock;

    unsigned extents[DIM];
    unsigned centers[DIM];
    for (int d = 0; d < DIM; d++) {
      assert(filter_rect.lo[d] == 0);
      extents[d] = filter_rect.hi[d] + 1;
      centers[d] = static_cast<coord_t>(extents[d] / 2);
    }
    unsigned tile[DIM];
    for (int d = DIM-1; d >= 0; d--) {
      // Make sure that each tile is at least double the size of the filter
      // so that we can get some savings in bandwidth needed 
      tile[d] = 2*centers[d];
      if (d == (DIM-1)) {
        // In order to maximize bandwidth, we want to make sure we're loading at
        // least 128B of contiguous memory along the last axis (row-major) of input
        const unsigned min_contig_elmts = 128 / sizeof(VAL);
        if ((tile[d] + 2*centers[d]) < min_contig_elmts)
          tile[d] = min_contig_elmts - 2*centers[d];
      } 
    }
    unsigned smem_size = sizeof(VAL);
    for (int d = 0; d < DIM; d++)
      smem_size *= (tile[d] + 2*centers[d]);
    if (smem_size <= max_smem_size) {
      // Case 1: Make the tile as big as possible so that it fits in shared memory
      // Try to keep it rectangular to minimize surface-to-volume ratio
      // and improve the reuse of data
      // If the current tile is less than half the shared memory in the SM then
      // decrease the upper bound so we can get 2 CTAs/SM
      bool halved = false;
      const unsigned half_smem = properties.sharedMemPerMultiprocessor / 2;
      if ((smem_size <= (half_smem)) && (half_smem < max_smem_size)) {
        max_smem_size = half_smem;
        halved = true;
      }
      if (DIM == 1) {
        // In this single case we can just solve for this directly
        unsigned elements = max_smem_size / sizeof(VAL);
        assert(elements > 2*centers[0]);
        assert(tile[0] < (elements - 2*centers[0]));
        tile[0] = elements - 2*centers[0];
        smem_size = (tile[0] + 2*centers[0]) * sizeof(VAL);
      } else {
        // Find the two smallest dimensions and increase one of them
        // until we hit the second smallest one or exceed max_smem_size
        bool all_same = true;
        while (true) {
          int d1 = DIM-1, d2 = -1;
          int t1 = tile[d1], t2 = 0;
          for (int d = DIM-2; d >= 0; d--) {
            if (tile[d] < t1) {
              d2 = d1;
              t2 = t1;
              d1 = d;
              t1 = tile[d];
            } else if ((d2 < 0) || (tile[d] < t2)) {
              d2 = d;
              t2 = tile[d];
            }
          }
          // If we ever get two dimensions of the same size then we know
          // that there is no smallest dimension so we can march all the
          // dimensions together at this point
          if (t1 == t2)
            break;
          // Solve for the max we can walk 
          unsigned pitch = sizeof(VAL);
          for (int d = 0; d < DIM; d++)
            if (d != d1)
              pitch *= (tile[d] + 2*centers[d]);
          unsigned elements = max_smem_size / pitch;
          assert(elements > 2*centers[d1]);
          assert(t1 < (elements - 2*centers[d1]));
          unsigned bound = elements - 2*centers[d1];
          if (bound < t2) {
            tile[d1] = bound;
            smem_size = pitch * (bound + 2*centers[d1]);
            all_same = false;
            break;
          } else {
            tile[d1] = t2;
            smem_size = pitch * (t2 + 2*centers[d1]);
          }
        }
        if (all_same) {
          // Step all the dimensions together until we hit
          // the shared memory upper bound we're targetting
          // This algorithm is in theory slow, but the max
          // amount of shared memory is small and the amount
          // of memory will grow polynomially in the number
          // of dimensions so it should converge quickly
          while (true) {
            unsigned next_smem_size = sizeof(VAL);
            for (int d = 0; d < DIM; d++)
              next_smem_size *= (tile[d] + 1 + 2*centers[d]);
            if (next_smem_size > max_smem_size) 
              break;
            smem_size = next_smem_size;
            for (int d = 0; d < DIM; d++)
              tile[d]++;
          }
        }
      }
      // At this point we've got the tile size that we're going to compute
      // and the amount of dynamic shared memory that we need
      // Compute the arguments needed for the kernel launch
      ConvolutionCase1Args<DIM> args;
      size_t blocks = 1;
      size_t tile_pitch = 1;
      unsigned input_pitch = 1;
      args.filter_volume = 1;
      for (int d = DIM-1; d >= 0; d--) {
        size_t blocks_along_dim =
          ((root_rect.hi[d] - root_rect.lo[d]) + tile[d]) / tile[d];
        args.grid_pitches[d] = FastDivmodU64(blocks);
        blocks *= blocks_along_dim;
        args.block_tiles[d] = tile[d];
        args.block_pitches[d] = FastDivmodU64(tile_pitch);
        tile_pitch *= tile[d];
        args.delta_lo[d] = centers[d];
        args.delta_hi[d] = tile[d] + centers[d] - 1;
        args.input_pitches[d] = FastDivmodU64(input_pitch);
        input_pitch *= (args.delta_lo[d] + args.delta_hi[d] + 1);
        args.filter_centers[d] = centers[d]; 
        args.filter_extents[d] = extents[d];
        args.filter_volume *= extents[d];
      }
      args.tile_volume = tile_pitch;
      args.input_volume = input_pitch;
      assert((input_pitch * sizeof(VAL)) == smem_size);
      if (halved) {
        if (tile_pitch < 512)
          convolution_case1a_kernel<VAL,DIM><<<blocks,tile_pitch,smem_size>>>(
              out, filter, in, root_rect, subrect, filter_rect, args);
        else
          convolution_case1a_kernel<VAL,DIM><<<blocks,512,smem_size>>>(
              out, filter, in, root_rect, subrect, filter_rect, args);
      } else {
        if (tile_pitch < 1024)
          convolution_case1b_kernel<VAL,DIM><<<blocks,tile_pitch,smem_size>>>(
              out, filter, in, root_rect, subrect, filter_rect, args);
        else
          convolution_case1b_kernel<VAL,DIM><<<blocks,1024,smem_size>>>(
              out, filter, in, root_rect, subrect, filter_rect, args);
      }
    } else {
#if 0
      // See if we can find one dimension that is not the last one to walk
      // over and still be able to fit the tile in shared memory, pick the
      // smallest dimension that still allows things to fit to minimize the
      // size of the pencils, break tiles with slower changing dimensions
      int pencil_dim = -1;
      for (int d = 0; d < (DIM-1); d++) {
        size_t alt_size = smem_size / (tile[d] + 2*centers[d]);
        // If it's still too big then skip it
        if (alt_size > smem_size)
          continue;
        if (pencil_dim >= 0) {
          if (centers[d] < centers[pencil_dim])
            pencil_dim = d;
        } else {
          pencil_dim = d;
        }
      }
      if (pencil_dim >= 0) {
        // Case 2: In this case the tile is what it is going to be, we
        // don't dare make it any bigger lest we risk the wrath of the
        // compiler spilling registers, especially for very large pencils
        // Figure out how many points we have and divide them among the
        // compute warps, then figure out how big the pencil arrays need
        // to be for each of the compute warps


      } else {
#endif
      {
        // Case 3: Punt and launch the dumb help-me-cache-god kernel
        // Figure out the tile size for the thread block. We want at
        // least 128B loads along the last dimension if possible. Then
        // round-robin powers of 2 onto the other dimensions until we 
        // get the tile to have as many threads as THREADS_PER_BLOCK.
        size_t limits[DIM];
        for (int d = 0; d < DIM; d++) {
          tile[d] = 1;
          limits[d] = root_rect.hi[d] - root_rect.lo[d] + 1;
        }
        // 2^5 == 32
        unsigned skip_dims = 0;
        for (int i = 0; i < 5; i++) {
          tile[DIM-1] *= 2;
          if (tile[DIM-1] >= limits[DIM-1]) {
            skip_dims |= (1 << (DIM-1));
            break;
          }
        }
        unsigned threads = tile[DIM-1];
        for (int i = 0; i < 5; i++) {
          for (int d = DIM-2; d >= 0; d--) {
            if (skip_dims & (1 << d))
              continue;
            tile[d] *= 2;
            threads *= 2;
            if (tile[d] >= limits[d]) {
              skip_dims |= (1 << d);
              continue;
            }
            if (threads == THREADS_PER_BLOCK)
              break;
          }
          if (threads == THREADS_PER_BLOCK)
            break;
        }
        while ((threads < THREADS_PER_BLOCK) &&
              (skip_dims != ((1 << (DIM+1)) - 1))) {
          for (int d = DIM-1; d >= 0; d--) {
            if (skip_dims & (1 << d))
              continue;
            tile[d] *= 2;
            threads *= 2;
            if (tile[d] >= limits[d]) {
              skip_dims |= (1 << d);
              continue;
            }
            if (threads == THREADS_PER_BLOCK)
              break;
          }
        }
        // should either not have enough points or
        // THREADS_PER_BLOCK should be a power of 2
        assert(threads <= THREADS_PER_BLOCK);
        // Compute the arguments needed to launch the kernel
        ConvolutionCase3Args<DIM> args;
        threads = 1;
        size_t blocks = 1;
        args.filter_volume = 1;
        for (int d = DIM-1; d >= 0; d--) {
          size_t blocks_along_dim =
            ((root_rect.hi[d] - root_rect.lo[d]) + tile[d]) / tile[d];
          args.grid_pitches[d] = FastDivmodU64(blocks);
          blocks *= blocks_along_dim;
          args.block_tiles[d] = tile[d];
          args.block_pitches[d] = FastDivmodU64(threads);
          threads *= tile[d];
          args.filter_centers[d] = centers[d]; 
          args.filter_extents[d] = extents[d];
          args.filter_volume *= extents[d];
        }
        convolution_case3_kernel<VAL,DIM><<<blocks,threads>>>(
            out, filter, in, root_rect, subrect, filter_rect, args);
      }
    }
  }
};

/*static*/ void ConvolveTask::gpu_variant(TaskContext& context)
{
  convolve_template<VariantKind::GPU>(context);
}

}  // namespace numpy
}  // namespace legate
