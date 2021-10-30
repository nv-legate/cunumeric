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

#include "cunumeric/divmod.h"
#include "cunumeric/cuda_help.h"
#include "cunumeric/convolution/convolve.h"
#include "cunumeric/convolution/convolve_template.inl"

namespace cunumeric {

using namespace Legion;
using namespace legate;

// Convolution should be able to hit FMA throughput limits
// on the GPU due to the amount of FLOPs needed to be performed
// given the amount of data loaded. This is especially true of
// larger convolution filters. In order to hit these limits though
// we need to make sure that the GPU is fed data appropriately.
// We have two different kernels to handle different sized filters.

// Small Tile Case
// In the small tile case, a reasonable tile input including the
// all the boundary values for a given filter tile can fit in the
// shared memory of the SM, allowing the threadblock to fully
// compute an entire tile of output points in a single pass.
// If the tile is small enough, we even try to get multiple CTAs/SM
// in order to better pipeline data loading with compute.

// Large Tile Case
// For inputs where the filter is very large and it is impossible
// to fit a reasonable sized tile into shared memory, we tile both
// the output and the filter and make multiple passes over the data
// to create reasonable sized input tiles that fit in shared memory.
// If possible we also attempt to tile for the L2 cache as well so
// that threadblocks walking through memory together can hopefully
// hit in the L2 more often than not when loading data

template <int DIM>
struct ConvolutionInitArgs {
 public:
  FastDivmodU64 pitches[DIM];
};

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, 4)
  convolution_init(const AccessorWO<VAL, DIM> out,
                   const Point<DIM> subrect_lo,
                   const ConvolutionInitArgs<DIM> args,
                   const size_t volume)
{
  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= volume) return;
  Point<DIM> point = subrect_lo;
#pragma unroll
  for (int d = 0; d < DIM; d++) point[d] += args.pitches[d].divmod(offset, offset);
  out[point] = VAL{0};
}

// We want to run the convolution kernel with as large a shared memory
// tile as possible to avoid duplicate loading of data and maximize
// compute intensity. Therefore we're always going to run with 1 CTA
// per SM, but we still want enough thread-level parallelism, so we
// set this to the maximum number of warps in a threadblock
// Note that a lot of this code assumes this is a power of 2
#define CONVOLUTION_THREADS 1024
// The idea behind THREAD_OUTPUTS is to figure out how many registers
// we will be willing to assign to hold the partial output accumulations
// in each thread without using too many registers. Every GPU (with one
// exception) has 64K 32-bit registers per SM. We key off that and want
// to allocate a quarter of those registers for holding the partial accumulations
// We assume here that sizeof(VAL) is a power of 2
#define THREAD_OUTPUTS(TYPE) 1
//((4/*bytes/reg*/ * ((65536/8)/CONVOLUTION_THREADS)/*regs/thread*/) / sizeof(TYPE))

template <int DIM, int POINTS>
struct ConvolutionLargeTileArgs {
  FastDivmod l1_output_tile_pitches[DIM];
  FastDivmod l1_input_pitches[DIM];
  FastDivmod l1_filter_pitches[DIM];
  FastDivmod l1_output_pitches[DIM];
  Point<DIM> l2_output_limits;
  Point<DIM, unsigned> point_offsets[POINTS];
  Point<DIM, unsigned> l2_output_tile;
  Point<DIM, unsigned> l2_filter_tile;
  Point<DIM, unsigned> l1_output_tile;
  Point<DIM, unsigned> l1_filter_tile;
  unsigned total_l2_outputs;
  unsigned total_l1_outputs;
  unsigned total_l1_filters;
  unsigned total_l1_points;
  unsigned l1_filter_points;
  unsigned l1_input_points;
  unsigned shared_input_offset;
  unsigned uniform_input_stride;
  unsigned shared_input_bound;
};

template <typename VAL, int DIM, int POINTS>
__global__ static void __launch_bounds__(CONVOLUTION_THREADS, 1)
  convolution_large_tile(const AccessorWO<VAL, DIM> out,
                         const AccessorRO<VAL, DIM> filter,
                         const AccessorRO<VAL, DIM> in,
                         const Rect<DIM> root_rect,
                         const Rect<DIM> subrect,
                         const Rect<DIM> l2_filter_rect,
                         const Point<DIM> l2_input_start,
                         const Point<DIM> l2_input_stop,
                         const Point<DIM> l1_input_start,
                         const Point<DIM, unsigned> zero,
                         const Point<DIM, unsigned> one,
                         const ConvolutionLargeTileArgs<DIM, POINTS> args)
{
  // Deal with compiler shared memory stupidity
  extern __shared__ uint8_t buffer[];
  // Technically this is illegal C++, but there's no other way to do it
  VAL* sharedmem = (VAL*)buffer;
  Point<DIM, unsigned> thread_offset;
  int offset = threadIdx.x;
#pragma unroll
  for (int d = 0; d < DIM; d++) thread_offset[d] = args.l1_output_pitches[d].divmod(offset, offset);
  Point<DIM> l2_output_offset = zero;
  for (unsigned l2_outidx = 0; l2_outidx < args.total_l2_outputs; l2_outidx++) {
    // Do a quick check here to see if all the inputs are contained for this tile
    // l2_input_start = subrect.lo + args.extents - l2_filter_rect.hi - one - centers
    // l2_input_stop = subrect.lo + l2_output_tile - one + args.extents - l2_filter_rect.lo - one -
    // centers
    const Rect<DIM> l2_input_rect(l2_input_start + l2_output_offset,
                                  l2_input_stop + l2_output_offset);
    const bool input_contained = root_rect.contains(l2_input_rect);
    // Iterate the L1 output tiles that this threadblock should compute for the L2 output
    for (unsigned l1_outidx = blockIdx.x; l1_outidx < args.total_l1_outputs;
         l1_outidx += gridDim.x) {
      Point<DIM, unsigned> l1_output_offset;
      offset = l1_outidx;
#pragma unroll
      for (int d = 0; d < DIM; d++)
        l1_output_offset[d] =
          args.l1_output_tile_pitches[d].divmod(offset, offset) * args.l1_output_tile[d];
      // Handle the boundary case where an L1 tile is not contained in the L2 tile
      // becasue the L2 tile is overlapping a boundary. Note this decisions is the
      // same for all the threads in the threadblock so no bad divergence
      bool output_contained = true;
#pragma unroll
      for (int d = 0; d < DIM; d++) {
        if ((subrect.lo[d] + l2_output_offset[d] + l1_output_offset[d]) <= subrect.hi[d]) continue;
        output_contained = false;
        break;
      }
      if (!output_contained) continue;
      // Initialize our point data
      VAL acc[POINTS];
#pragma unroll
      for (int p = 0; p < POINTS; p++) acc[p] = VAL{0};
      // Iterate over the l1 filter tiles
      Point<DIM, unsigned> l1_filter_offset = zero;
      for (unsigned l1_fidx = 0; l1_fidx < args.total_l1_filters; l1_fidx++) {
        // Wait for any previous readers to be done
        __syncthreads();
// Load the filter into shared memory
// Unroll this a few times to get some memory level parallelims
#pragma unroll 4
        for (unsigned fidx = threadIdx.x; fidx < args.l1_filter_points; fidx += blockDim.x) {
          Point<DIM> filter_point = l2_filter_rect.lo + l1_filter_offset;
          offset                  = fidx;
#pragma unroll
          for (int d = 0; d < DIM; d++)
            filter_point[d] += args.l1_filter_pitches[d].divmod(offset, offset);
          if (l2_filter_rect.contains(filter_point))
            sharedmem[fidx] = filter[filter_point];
          else
            sharedmem[fidx] = VAL{0};
        }
        // Load the input into shared memory
        // Compute the input start point
        // input_start = subrect.lo + extents - l2_filter_rect.lo - l1_filter_tile - centers
        Point<DIM> input_start = l1_input_start + l2_output_offset + l1_output_offset;
        input_start -= l1_filter_offset;
// Unroll this a few times to get some memory level parallelism
#pragma unroll 4
        for (unsigned idx = threadIdx.x; idx < args.l1_input_points; idx += blockDim.x) {
          Point<DIM> input_point = input_start;
          offset                 = idx;
#pragma unroll
          for (int d = 0; d < DIM; d++)
            input_point[d] += args.l1_input_pitches[d].divmod(offset, offset);
          if (input_contained || root_rect.contains(input_point))
            sharedmem[args.shared_input_offset + idx] = in[input_point];
          else
            sharedmem[args.shared_input_offset + idx] = VAL{0};
        }
        // Wait for everything to be loaded into shared memory
        __syncthreads();
        // Iterate the points in the filter
        // We can safely iterate all the filter points and input points
        // because we wrote zeros into shared memory for everything that
        // was out of bounds
        Point<DIM, unsigned> filter_point = zero;
        if (args.uniform_input_stride) {
          // Each point is a constant offset in shared from the others
          unsigned input_offset = args.shared_input_offset;
#pragma unroll
          for (int d = 0; d < DIM; d++)
            input_offset +=
              args.l1_input_pitches[d].divisor * (thread_offset[d] + args.l1_filter_tile[d] - 1);
          if (args.shared_input_bound) {
            for (unsigned fidx = 0; fidx < args.l1_filter_points; fidx++) {
              // Use shared memory broadcasting functionality to avoid bank conflicts
              const VAL filter_value = sharedmem[fidx];
              unsigned point_offset  = input_offset;
#pragma unroll
              for (int p = 0; p < POINTS; p++) {
                if (args.shared_input_bound <= point_offset) break;
                acc[p] = acc[p] + filter_value * sharedmem[point_offset];
                point_offset += args.uniform_input_stride;
              }
// Step to the next filter point and update the input stride
#pragma unroll
              for (int d = DIM - 1; d >= 0; d--) {
                filter_point[d]++;
                input_offset -= args.l1_input_pitches[d].divisor;
                if (filter_point[d] == args.l1_filter_tile[d]) {
                  input_offset += args.l1_filter_tile[d] * args.l1_input_pitches[d].divisor;
                  filter_point[d] = 0;
                } else {
                  break;
                }
              }
            }
          } else {
            for (unsigned fidx = 0; fidx < args.l1_filter_points; fidx++) {
              // Use shared memory broadcasting functionality to avoid bank conflicts
              const VAL filter_value = sharedmem[fidx];
              unsigned point_offset  = input_offset;
#pragma unroll
              for (int p = 0; p < POINTS; p++) {
                acc[p] = acc[p] + filter_value * sharedmem[point_offset];
                point_offset += args.uniform_input_stride;
              }
// Step to the next filter point and update the input stride
#pragma unroll
              for (int d = DIM - 1; d >= 0; d--) {
                filter_point[d]++;
                input_offset -= args.l1_input_pitches[d].divisor;
                if (filter_point[d] == args.l1_filter_tile[d]) {
                  input_offset += args.l1_filter_tile[d] * args.l1_input_pitches[d].divisor;
                  filter_point[d] = 0;
                } else {
                  break;
                }
              }
            }
          }
        } else {
          // Need to compute the input offset uniquely for each point
          Point<DIM, unsigned> input_point = thread_offset + args.l1_filter_tile - one;
          unsigned point_offsets[POINTS];
#pragma unroll
          for (int p = 0; p < POINTS; p++) {
            point_offsets[p] = args.shared_input_offset;
#pragma unroll
            for (int d = 0; d < DIM; d++)
              point_offsets[p] +=
                (input_point[d] + args.point_offsets[p][d]) * args.l1_input_pitches[d].divisor;
          }
          unsigned filter_offset = 0;
          if (args.shared_input_bound) {
            for (unsigned fidx = 0; fidx < args.l1_filter_points; fidx++) {
              // Use shared memory broadcasting functionality to avoid bank conflicts
              const VAL filter_value = sharedmem[fidx];
#pragma unroll
              for (int p = 0; p < POINTS; p++) {
                unsigned point_offset = point_offsets[p] - filter_offset;
                if (args.shared_input_bound <= point_offset) continue;
                acc[p] = acc[p] + filter_value * sharedmem[point_offset];
              }
// Step to the next filter point
#pragma unroll
              for (int d = DIM - 1; d >= 0; d--) {
                filter_point[d]++;
                filter_offset += args.l1_input_pitches[d].divisor;
                if (filter_point[d] == args.l1_filter_tile[d]) {
                  filter_offset -= args.l1_filter_tile[d] * args.l1_input_pitches[d].divisor;
                  filter_point[d] = 0;
                } else {
                  break;
                }
              }
            }
          } else {
            for (unsigned fidx = 0; fidx < args.l1_filter_points; fidx++) {
              // Use shared memory broadcasting functionality to avoid bank conflicts
              const VAL filter_value = sharedmem[fidx];
#pragma unroll
              for (int p = 0; p < POINTS; p++) {
                unsigned point_offset = point_offsets[p] - filter_offset;
                acc[p]                = acc[p] + filter_value * sharedmem[point_offset];
              }
// Step to the next filter point
#pragma unroll
              for (int d = DIM - 1; d >= 0; d--) {
                filter_point[d]++;
                filter_offset += args.l1_input_pitches[d].divisor;
                if (filter_point[d] == args.l1_filter_tile[d]) {
                  filter_offset -= args.l1_filter_tile[d] * args.l1_input_pitches[d].divisor;
                  filter_point[d] = 0;
                } else {
                  break;
                }
              }
            }
          }
        }
// Step to the next L1 filter tile
#pragma unroll
        for (int d = DIM - 1; d >= 0; d--) {
          l1_filter_offset[d] += args.l1_filter_tile[d];
          if (args.l2_filter_tile[d] <= l1_filter_offset[d])
            l1_filter_offset[d] = 0;
          else
            break;
        }
      }
      // Now we can stream our accumulators back to the output
      Point<DIM> output = subrect.lo + l2_output_offset + (l1_output_offset + thread_offset);
      if (input_contained) {
        // If the input was contained, then so is the output
        if (args.total_l1_points) {
          unsigned index = threadIdx.x;
#pragma unroll
          for (int p = 0; p < POINTS; p++) {
            if (args.total_l1_points <= index) break;
            VAL* ptr = out.ptr(output + args.point_offsets[p]);
            // Make sure we don't pollute the L2 cache
            VAL value = load_streaming<VAL>(ptr);
            store_streaming<VAL>(ptr, value + acc[p]);
            index += blockDim.x;
          }
        } else {
#pragma unroll
          for (int p = 0; p < POINTS; p++) {
            VAL* ptr = out.ptr(output + args.point_offsets[p]);
            // Make sure we don't pollute the L2 cache
            VAL value = load_streaming<VAL>(ptr);
            store_streaming<VAL>(ptr, value + acc[p]);
          }
        }
      } else {
        // Input was not contained, so the output might not be either, do checks
        if (args.total_l1_points) {
          unsigned index = threadIdx.x;
#pragma unroll
          for (int p = 0; p < POINTS; p++) {
            if (args.total_l1_points <= index) break;
            Point<DIM> point = output + args.point_offsets[p];
            if (!subrect.contains(point)) break;
            VAL* ptr = out.ptr(point);
            // Make sure we don't pollute the L2 cache
            VAL value = load_streaming<VAL>(ptr);
            store_streaming<VAL>(ptr, value + acc[p]);
            index += blockDim.x;
          }
        } else {
#pragma unroll
          for (int p = 0; p < POINTS; p++) {
            Point<DIM> point = output + args.point_offsets[p];
            if (!subrect.contains(point)) continue;
            VAL* ptr = out.ptr(point);
            // Make sure we don't pollute the L2 cache
            VAL value = load_streaming<VAL>(ptr);
            store_streaming<VAL>(ptr, value + acc[p]);
          }
        }
      }
    }
// Step to the next output tile
#pragma unroll
    for (int d = DIM - 1; d >= 0; d--) {
      l2_output_offset[d] += args.l2_output_tile[d];
      if (args.l2_output_limits[d] <= l2_output_offset[d])
        l2_output_offset[d] = 0;
      else
        break;
    }
  }
}

template <int DIM>
struct ConvolutionSmallTileArgs {
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

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(512, 2)
  convolution_small_tile1(const AccessorWO<VAL, DIM> out,
                          const AccessorRO<VAL, DIM> filter,
                          const AccessorRO<VAL, DIM> in,
                          const Rect<DIM> root_rect,
                          const Rect<DIM> subrect,
                          const Rect<DIM> filter_rect,
                          const ConvolutionSmallTileArgs<DIM> args)
{
  // Deal with compiler shared memory stupidity
  extern __shared__ uint8_t buffer[];
  // Technically this illegal C++, but there's no other way to do it
  VAL* input = (VAL*)buffer;
  // Compute the origin point of the block
  size_t offset          = blockIdx.x;
  Point<DIM> block_point = subrect.lo;
#pragma unroll
  for (int d = 0; d < DIM; d++)
    block_point[d] += args.grid_pitches[d].divmod(offset, offset) * args.block_tiles[d];
  // Load in the shared memory for this block
  Point<DIM> tile_point;
  const Rect<DIM> input_bounds(block_point - args.delta_lo, block_point + args.delta_hi);
  const bool input_contained = root_rect.contains(input_bounds);
  if (input_contained) {
// All the points are contained, so no need for point-wise tests
// Unroll this four times to try to pipeline loads
#pragma unroll 4
    for (unsigned idx = threadIdx.x; idx < args.input_volume; idx += blockDim.x) {
      offset = idx;
#pragma unroll
      for (int d = 0; d < DIM; d++) tile_point[d] = args.input_pitches[d].divmod(offset, offset);
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
      for (int d = 0; d < DIM; d++) tile_point[d] = args.input_pitches[d].divmod(offset, offset);
      if (!root_rect.contains(input_bounds.lo + tile_point)) continue;
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
      out_point[d]  = block_point[d] + tile_point[d];
    }
    if (!subrect.contains(out_point)) continue;
#pragma unroll
    for (int d = 0; d < DIM; d++) f_coords[d] = 0;
    VAL acc{0};
    for (unsigned idx = 0; idx < args.filter_volume; idx++) {
#pragma unroll
      for (int d = 0; d < DIM; d++)
        in_point[d] = out_point[d] + f_coords[d] - args.filter_centers[d];
      if (input_contained || root_rect.contains(in_point)) {
        offset = 0;
#pragma unroll
        for (int d = 0; d < DIM; d++)
          offset += (tile_point[d] + f_coords[d]) * args.input_pitches[d].divisor;
#pragma unroll
        for (int d = 0; d < DIM; d++) filter_point[d] = args.filter_extents[d] - f_coords[d] - 1;
        acc = acc + input[offset] * filter[filter_point];
      }
// Step the filter coordinates
#pragma unroll
      for (int d = DIM - 1; d >= 0; d--) {
        f_coords[d]++;
        if (f_coords[d] == args.filter_extents[d])
          f_coords[d] = 0;
        else
          break;
      }
    }
    store_streaming(out.ptr(out_point), acc);
  }
}

// This version of the kernel is identical to the one above but with
// different launch bounds to handle a bigger CTA with more shared memory
template <typename VAL, int DIM>
__global__ static void __launch_bounds__(1024, 1)
  convolution_small_tile2(const AccessorWO<VAL, DIM> out,
                          const AccessorRO<VAL, DIM> filter,
                          const AccessorRO<VAL, DIM> in,
                          const Rect<DIM> root_rect,
                          const Rect<DIM> subrect,
                          const Rect<DIM> filter_rect,
                          const ConvolutionSmallTileArgs<DIM> args)
{
  // Deal with compiler shared memory stupidity
  extern __shared__ uint8_t buffer[];
  // Technically this illegal C++, but there's no other way to do it
  VAL* input = (VAL*)buffer;
  // Compute the origin point of the block
  size_t offset          = blockIdx.x;
  Point<DIM> block_point = subrect.lo;
#pragma unroll
  for (int d = 0; d < DIM; d++)
    block_point[d] += args.grid_pitches[d].divmod(offset, offset) * args.block_tiles[d];
  // Load in the shared memory for this block
  Point<DIM> tile_point;
  const Rect<DIM> input_bounds(block_point - args.delta_lo, block_point + args.delta_hi);
  const bool input_contained = root_rect.contains(input_bounds);
  if (input_contained) {
// All the points are contained, so no need for point-wise tests
// Unroll this four times to try to pipeline loads
#pragma unroll 4
    for (unsigned idx = threadIdx.x; idx < args.input_volume; idx += blockDim.x) {
      offset = idx;
#pragma unroll
      for (int d = 0; d < DIM; d++) tile_point[d] = args.input_pitches[d].divmod(offset, offset);
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
      for (int d = 0; d < DIM; d++) tile_point[d] = args.input_pitches[d].divmod(offset, offset);
      if (!root_rect.contains(input_bounds.lo + tile_point)) continue;
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
      out_point[d]  = block_point[d] + tile_point[d];
    }
    if (!subrect.contains(out_point)) continue;
#pragma unroll
    for (int d = 0; d < DIM; d++) f_coords[d] = 0;
    VAL acc{0};
    for (unsigned idx = 0; idx < args.filter_volume; idx++) {
#pragma unroll
      for (int d = 0; d < DIM; d++)
        in_point[d] = out_point[d] + f_coords[d] - args.filter_centers[d];
      if (input_contained || root_rect.contains(in_point)) {
        offset = 0;
#pragma unroll
        for (int d = 0; d < DIM; d++)
          offset += (tile_point[d] + f_coords[d]) * args.input_pitches[d].divisor;
#pragma unroll
        for (int d = 0; d < DIM; d++) filter_point[d] = args.filter_extents[d] - f_coords[d] - 1;
        acc = acc + input[offset] * filter[filter_point];
      }
// Step the filter coordinates
#pragma unroll
      for (int d = DIM - 1; d >= 0; d--) {
        f_coords[d]++;
        if (f_coords[d] == args.filter_extents[d])
          f_coords[d] = 0;
        else
          break;
      }
    }
    store_streaming(out.ptr(out_point), acc);
  }
}

template <typename VAL, int DIM>
__host__ static inline void launch_small_tile_kernel(AccessorWO<VAL, DIM> out,
                                                     AccessorRO<VAL, DIM> filter,
                                                     AccessorRO<VAL, DIM> in,
                                                     const Rect<DIM>& root_rect,
                                                     const Rect<DIM>& subrect,
                                                     const Rect<DIM>& filter_rect,
                                                     const cudaDeviceProp& properties,
                                                     const unsigned extents[DIM],
                                                     const unsigned centers[DIM],
                                                     Point<DIM>& tile,
                                                     unsigned smem_size,
                                                     size_t max_smem_size)
{
  // Make the tile as big as possible so that it fits in shared memory
  // Try to keep it rectangular to minimize surface-to-volume ratio
  // and improve the reuse of data
  // If the current tile is less than half the shared memory in the SM then
  // decrease the upper bound so we can get 2 CTAs/SM
  bool halved              = false;
  const unsigned half_smem = properties.sharedMemPerMultiprocessor / 2;
  if ((smem_size <= (half_smem)) && (half_smem < max_smem_size)) {
    max_smem_size = half_smem;
    halved        = true;
  }
  Point<DIM> padding;
  for (int d = 0; d < DIM; d++) padding[d] = 2 * centers[d];
  Point<DIM> bounds = subrect.hi - subrect.lo + Point<DIM>::ONES();
  smem_size         = roundup_tile<VAL, DIM>(tile, bounds, padding, max_smem_size);
  // At this point we've got the tile size that we're going to compute
  // and the amount of dynamic shared memory that we need
  // Compute the arguments needed for the kernel launch
  ConvolutionSmallTileArgs<DIM> args;
  size_t blocks        = 1;
  size_t tile_pitch    = 1;
  unsigned input_pitch = 1;
  args.filter_volume   = 1;
  for (int d = DIM - 1; d >= 0; d--) {
    size_t blocks_along_dim = ((subrect.hi[d] - subrect.lo[d]) + tile[d]) / tile[d];
    args.grid_pitches[d]    = FastDivmodU64(blocks);
    blocks *= blocks_along_dim;
    args.block_tiles[d]   = tile[d];
    args.block_pitches[d] = FastDivmodU64(tile_pitch);
    tile_pitch *= tile[d];
    args.delta_lo[d]      = centers[d];
    args.delta_hi[d]      = tile[d] + centers[d] - 1;
    args.input_pitches[d] = FastDivmodU64(input_pitch);
    input_pitch *= (args.delta_lo[d] + args.delta_hi[d] + 1);
    args.filter_centers[d] = centers[d];
    args.filter_extents[d] = extents[d];
    args.filter_volume *= extents[d];
  }
  args.tile_volume  = tile_pitch;
  args.input_volume = input_pitch;
  assert((input_pitch * sizeof(VAL)) == smem_size);
  if (halved) {
    if (tile_pitch < 512)
      convolution_small_tile1<VAL, DIM>
        <<<blocks, tile_pitch, smem_size>>>(out, filter, in, root_rect, subrect, filter_rect, args);
    else
      convolution_small_tile1<VAL, DIM>
        <<<blocks, 512, smem_size>>>(out, filter, in, root_rect, subrect, filter_rect, args);
  } else {
    if (tile_pitch < 1024)
      convolution_small_tile2<VAL, DIM>
        <<<blocks, tile_pitch, smem_size>>>(out, filter, in, root_rect, subrect, filter_rect, args);
    else
      convolution_small_tile2<VAL, DIM>
        <<<blocks, 1024, smem_size>>>(out, filter, in, root_rect, subrect, filter_rect, args);
  }
}

template <LegateTypeCode CODE, int DIM>
struct ConvolveImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  __host__ void operator()(AccessorWO<VAL, DIM> out,
                           AccessorRO<VAL, DIM> filter,
                           AccessorRO<VAL, DIM> in,
                           const Rect<DIM>& root_rect,
                           const Rect<DIM>& subrect,
                           const Rect<DIM>& filter_rect) const
  {
    constexpr int THREADVALS = THREAD_OUTPUTS(VAL);
    // Get the maximum amount of shared memory per threadblock
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp properties;
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    size_t max_smem_size = properties.sharedMemPerBlockOptin;

    // Only need to do these calls the first time on each device so
    // we use a bit mask to track which devices we've done it for
    static unsigned long long mask = 0;
    if (!(mask & (1 << device))) {
      if (properties.sharedMemPerBlock < max_smem_size) {
        CHECK_CUDA(cudaFuncSetAttribute(convolution_small_tile1<VAL, DIM>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        max_smem_size));
        CHECK_CUDA(cudaFuncSetAttribute(convolution_small_tile2<VAL, DIM>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        max_smem_size));
        CHECK_CUDA(cudaFuncSetAttribute(convolution_large_tile<VAL, DIM, THREADVALS>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        max_smem_size));
      }
      if (sizeof(VAL) >= 8) {
        // Only need to set this on the first invocation
        CHECK_CUDA(cudaFuncSetSharedMemConfig(convolution_small_tile1<VAL, DIM>,
                                              cudaSharedMemBankSizeEightByte));
        CHECK_CUDA(cudaFuncSetSharedMemConfig(convolution_small_tile2<VAL, DIM>,
                                              cudaSharedMemBankSizeEightByte));
        CHECK_CUDA(cudaFuncSetSharedMemConfig(convolution_large_tile<VAL, DIM, THREADVALS>,
                                              cudaSharedMemBankSizeEightByte));
      }
      // Make sure we have enough bits for every device
      assert(device < (8 * sizeof(mask)));
      // Make sure not to race with updates from other GPUs
      __sync_fetch_and_add(&mask, (1 << device));
    }
    unsigned extents[DIM];
    unsigned centers[DIM];
    for (int d = 0; d < DIM; d++) {
      assert(filter_rect.lo[d] == 0);
      extents[d] = filter_rect.hi[d] + 1;
      centers[d] = static_cast<coord_t>(extents[d] / 2);
    }
    Point<DIM> tile;
    for (int d = DIM - 1; d >= 0; d--) {
      // Make sure that each tile is at least double the size of the filter
      // so that we can get some savings in bandwidth needed
      tile[d] = 2 * centers[d];
      if (d == (DIM - 1)) {
        // In order to maximize bandwidth, we want to make sure we're loading at
        // least 128B of contiguous memory along the last axis (row-major) of input
        const unsigned min_contig_elmts = 128 / sizeof(VAL);
        if ((tile[d] + 2 * centers[d]) < min_contig_elmts)
          tile[d] = min_contig_elmts - 2 * centers[d];
      }
    }
    unsigned smem_size = sizeof(VAL);
    for (int d = 0; d < DIM; d++) smem_size *= (tile[d] + 2 * centers[d]);
    if (smem_size <= max_smem_size) {
      // Small tile case:
      launch_small_tile_kernel<VAL, DIM>(out,
                                         filter,
                                         in,
                                         root_rect,
                                         subrect,
                                         filter_rect,
                                         properties,
                                         extents,
                                         centers,
                                         tile,
                                         smem_size,
                                         max_smem_size);
    } else {
      // Large tile case:
      // If we're going to do this, we need to initialize the output to zeros
      // so we can kick that off to the GPU while we figure out how to launch
      // the rest of the kernels to do the convolution
      size_t strides[DIM];
      VAL* out_ptr = out.ptr(subrect, strides);
      // Check to see if the output is dense
      bool out_dense   = true;
      size_t out_pitch = 1;
      for (int d = DIM - 1; d >= 0; d--) {
        if (strides[d] != out_pitch) {
          out_dense = false;
          break;
        }
        out_pitch *= strides[d];
      }
      if (out_dense) {
        size_t bytes = sizeof(VAL) * out_pitch;
        CHECK_CUDA(cudaMemsetAsync(out_ptr, 0, bytes));
      } else {
        out_pitch = 1;
        ConvolutionInitArgs<DIM> args;
        for (int d = DIM - 1; d >= 0; d--) {
          args.pitches[d] = FastDivmodU64(out_pitch);
          out_pitch *= (subrect.hi[d] - subrect.lo[d] + 1);
        }
        size_t blocks = (out_pitch + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        convolution_init<VAL, DIM><<<blocks, THREADS_PER_BLOCK>>>(out, subrect.lo, args, out_pitch);
      }
      // Figure out the shape of the L1 output tile based on the number of
      // points that we can fit into registers
      Point<DIM> l1_output_tile;
      const unsigned max_l1_output_volume = CONVOLUTION_THREADS * THREADVALS;
      // Make sure the max_l1_output_volume doesn't consume more than half of shared memory
      unsigned target_l1_output_volume = max_l1_output_volume;
      while ((max_smem_size / 2) < (target_l1_output_volume * sizeof(VAL)))
        target_l1_output_volume /= 2;
      const Point<DIM> output_bounds = subrect.hi - subrect.lo + Point<DIM>::ONES();
      const unsigned l1_output_volume =
        compute_output_tile<VAL, DIM>(l1_output_tile,
                                      output_bounds,
                                      128 /*cache line size*/ / sizeof(VAL),
                                      target_l1_output_volume);
      // At this point we've got our output tile, compute how big a filter
      // tile we can make and still fit both the filter tile and the
      // input tile into the maximum amount of shared memory for this GPU
      Point<DIM> l1_filter_tile;
      const Point<DIM> filter_bounds = filter_rect.hi - filter_rect.lo + Point<DIM>::ONES();
      unsigned dynamic_smem =
        compute_filter_tile<VAL, DIM>(l1_filter_tile, filter_bounds, l1_output_tile, max_smem_size);
      unsigned input_smem_offset = 1;
      for (int d = 0; d < DIM; d++) input_smem_offset *= l1_filter_tile[d];
      // Tile the number of SMs on this GPU to compute the shape of the
      // L2 output tile for this kernel
      // We assume here that the number of SMs is easily factorable
      // into primes of 2, 3, and 5. It would be strange if we have a
      // GPU with a number of SMs these days that can't be factored
      // this way. If we do report a warning.
      unsigned l2_tiles[DIM];
      for (int d = 0; d < DIM; d++) l2_tiles[d] = 1;
      if (DIM > 1) {
        unsigned twos = 0, threes = 0, fives = 0;
        unsigned remainder = properties.multiProcessorCount;
        while ((remainder > 1) && ((remainder % 2) == 0)) {
          twos++;
          remainder /= 2;
        }
        while ((remainder > 1) && ((remainder % 3) == 0)) {
          threes++;
          remainder /= 3;
        }
        while ((remainder > 1) && ((remainder % 5) == 0)) {
          fives++;
          remainder /= 5;
        }
        if (remainder > 1) {
          fprintf(stdout,
                  "WARNING: %d is an unusual number of SMs "
                  "for GPU convolution. Please report your GPU kind and "
                  "the number of SMs in a Legate NumPy issue.",
                  properties.multiProcessorCount);
          l2_tiles[DIM - 1] = remainder;
        }
        for (unsigned idx = 0; idx < fives; idx++) {
          int smallest = 0;
          for (int d = 1; d < DIM; d++) {
            if (l2_tiles[smallest] < l2_tiles[d]) continue;
            smallest = d;
          }
          l2_tiles[smallest] *= 5;
        }
        for (unsigned idx = 0; idx < threes; idx++) {
          int smallest = 0;
          for (int d = 1; d < DIM; d++) {
            if (l2_tiles[smallest] < l2_tiles[d]) continue;
            smallest = d;
          }
          l2_tiles[smallest] *= 3;
        }
        for (unsigned idx = 0; idx < twos; idx++) {
          int smallest = 0;
          for (int d = 1; d < DIM; d++) {
            if (l2_tiles[smallest] < l2_tiles[d]) continue;
            smallest = d;
          }
          l2_tiles[smallest] *= 2;
        }
      } else {
        l2_tiles[0] = properties.multiProcessorCount;
      }
      // Now that we've got a tiling of the l1 output blocks across
      // the SMs compute how big it is in memory and see if it is less
      // than a quarter of the L2 cache so we can block for the L2
      Point<DIM> l2_output_tile;
      size_t l2_output_tile_size = sizeof(VAL);
      for (int d = 0; d < DIM; d++) {
        l2_output_tile[d] = l2_tiles[d] * l1_output_tile[d];
        l2_output_tile_size *= l2_output_tile[d];
      }
      Point<DIM> l2_filter_tile;
      size_t total_l2_filters = 1;
      if (l2_output_tile_size <= (properties.l2CacheSize / 4)) {
        for (int d = 0; d < DIM; d++) l2_filter_tile[d] = 1;
        // Compute the L2 filter tile size so that the L2 filter and the
        // corresponding L2 input tile will fit in the L2 cache
        compute_filter_tile<VAL, DIM>(
          l2_filter_tile, filter_bounds, l2_output_tile, 3 * properties.l2CacheSize / 4);
        for (int d = 0; d < DIM; d++)
          total_l2_filters *= (filter_bounds[d] + l2_filter_tile[d] - 1) / l2_filter_tile[d];
      } else {
        // It's likely this tile is too big to block for the L2 cache
        // so we're not going to bother blocking for the L2 and just
        // run everything out of the framebuffer memory. The upside is
        // that we'll only need to make a single pass over the input
        for (int d = 0; d < DIM; d++) l2_filter_tile[d] = filter_rect.hi[d] - filter_rect.lo[d] + 1;
      }
      // Construct the arguments for the kernel launches
      ConvolutionLargeTileArgs<DIM, THREADVALS> args;
      int pitch = 1;
      for (int d = DIM - 1; d >= 0; d--) {
        args.l1_input_pitches[d] = FastDivmod(pitch);
        pitch *= (l1_output_tile[d] + 2 * (l1_filter_tile[d] / 2));
      }
      pitch = 1;
      for (int d = DIM - 1; d >= 0; d--) {
        args.l1_filter_pitches[d] = FastDivmod(pitch);
        pitch *= l1_filter_tile[d];
      }
      pitch = 1;
      for (int d = DIM - 1; d >= 0; d--) {
        args.l1_output_pitches[d] = FastDivmod(pitch);
        pitch *= l1_output_tile[d];
      }
      args.l2_output_tile      = l2_output_tile;
      args.l2_filter_tile      = l2_filter_tile;
      args.l1_output_tile      = l1_output_tile;
      args.l1_filter_tile      = l1_filter_tile;
      args.l2_output_limits    = output_bounds;
      args.shared_input_offset = input_smem_offset;
      args.total_l2_outputs    = 1;
      args.total_l1_outputs    = 1;
      args.total_l1_filters    = 1;
      args.l1_filter_points    = 1;
      args.l1_input_points     = 1;
      pitch                    = 1;
      for (int d = DIM - 1; d >= 0; d--) {
        args.total_l2_outputs *= (output_bounds[d] + l2_output_tile[d] - 1) / l2_output_tile[d];
        args.l1_output_tile_pitches[d] = FastDivmod(pitch);
        pitch *= (l2_output_tile[d] + l1_output_tile[d] - 1) / l1_output_tile[d];
        args.total_l1_filters *= (l2_filter_tile[d] + l1_filter_tile[d] - 1) / l1_filter_tile[d];
        args.l1_filter_points *= l1_filter_tile[d];
        args.l1_input_points *= (l1_output_tile[d] + 2 * (l1_filter_tile[d] / 2));
      }
      args.total_l1_outputs = pitch;
      // Figure out how to tile the points across the l1_output_tile
      if (DIM > 1) {
        unsigned regsteps[DIM];
        for (int d = 0; d < DIM; d++) regsteps[d] = 0;
        unsigned remainder = THREADVALS;
        // Handle the case here where we aren't going to use all
        // the points in the registers so we need to scale back
        if (l1_output_volume < max_l1_output_volume) {
          assert((max_l1_output_volume % l1_output_volume) == 0);
          remainder /= (max_l1_output_volume / l1_output_volume);
          if (remainder == 0) remainder = 1;
        }
        for (int d = 0; d < DIM; d++) {
          if (remainder == 1) {
            regsteps[d] = l1_output_tile[d];
          } else if (remainder <= l1_output_tile[d]) {
            // All powers of two so should always divide
            assert((l1_output_tile[d] % remainder) == 0);
            regsteps[d] = l1_output_tile[d] / remainder;
            remainder   = 1;
          } else {
            // All powers of two so should always divide
            assert((remainder % l1_output_tile[d]) == 0);
            regsteps[d] = 1;
            remainder /= l1_output_tile[d];
          }
        }
        assert(remainder == 1);
        Point<DIM, unsigned> offset = Point<DIM, unsigned>::ZEROES();
        for (int p = 0; p < THREADVALS; p++) {
          args.point_offsets[p] = offset;
          // Step to the next offset
          for (int d = DIM - 1; d >= 0; d--) {
            offset[d] += regsteps[d];
            if (offset[d] == l1_output_tile[d]) {
              if ((d == 0) && (p != (THREADVALS - 1)))
                // Allow overflow in this case to handle the case
                // where we have more points than we need for the l1 output tile
                assert(l1_output_volume < max_l1_output_volume);
              else
                offset[d] = 0;
            } else
              break;
          }
        }
        args.uniform_input_stride = regsteps[0] * args.l1_input_pitches[0].divisor;
        // Check to make sure this is the uniform input stride case
        for (int d = 1; d < DIM; d++) {
          if (regsteps[d] == l1_output_tile[d]) continue;
          args.uniform_input_stride = 0;
          break;
        }
      } else {
        assert(THREADVALS <= l1_output_tile[0]);
        unsigned remainder = THREADVALS;
        // Handle the case here where we aren't going to use all
        // the points in the registers so we need to scale back
        if (l1_output_volume < max_l1_output_volume) {
          assert((max_l1_output_volume % l1_output_volume) == 0);
          remainder /= (max_l1_output_volume / l1_output_volume);
          if (remainder == 0) remainder = 1;
        }
        assert((l1_output_tile[0] % remainder) == 0);
        unsigned regstep = l1_output_tile[0] / remainder;
        for (int p = 0; p < THREADVALS; p++) args.point_offsets[p][0] = p * regstep;
        args.uniform_input_stride = regstep * args.l1_input_pitches[0].divisor;
      }
      if (l1_output_volume < max_l1_output_volume) {
        args.shared_input_bound = dynamic_smem / sizeof(VAL);
        args.total_l1_points    = l1_output_volume;
      } else {
        args.shared_input_bound = 0;
        args.total_l1_points    = 0;
      }
      // Launch as many kernels as we need to walk over the entire filter
      // Given the L2 filter tile that we came up with
      const Point<DIM, unsigned> zero = Point<DIM, unsigned>::ZEROES();
      const Point<DIM, unsigned> one  = Point<DIM, unsigned>::ONES();
      if (total_l2_filters > 1) {
        Point<DIM> l2_filter_lo = filter_rect.lo;
        for (unsigned idx = 0; idx < total_l2_filters; idx++) {
          Rect<DIM> l2_filter_rect(l2_filter_lo, l2_filter_lo + l2_filter_tile - one);
          l2_filter_rect = l2_filter_rect.intersection(filter_rect);
          const Point<DIM> l1_input_start =
            subrect.lo + Point<DIM>(extents) - l2_filter_lo - l1_filter_tile - Point<DIM>(centers);
          const Point<DIM> l2_input_start =
            subrect.lo + Point<DIM>(extents) - l2_filter_rect.hi - one - Point<DIM>(centers);
          const Point<DIM> l2_input_stop = subrect.lo + l2_output_tile - one + Point<DIM>(extents) -
                                           l2_filter_rect.lo - one - Point<DIM>(centers);
          convolution_large_tile<VAL, DIM, THREADVALS>
            <<<properties.multiProcessorCount, CONVOLUTION_THREADS, dynamic_smem>>>(out,
                                                                                    filter,
                                                                                    in,
                                                                                    root_rect,
                                                                                    subrect,
                                                                                    l2_filter_rect,
                                                                                    l2_input_start,
                                                                                    l2_input_stop,
                                                                                    l1_input_start,
                                                                                    zero,
                                                                                    one,
                                                                                    args);
          // Step to the next filter
          for (int d = DIM - 1; d >= 0; d--) {
            l2_filter_lo[d] += l2_filter_tile[d];
            if (filter_rect.hi[d] < l2_filter_lo[d])
              l2_filter_lo[d] = filter_rect.lo[d];
            else
              break;
          }
        }
      } else {
        assert(total_l2_filters == 1);
        const Point<DIM> l1_input_start =
          subrect.lo + Point<DIM>(extents) - filter_rect.lo - l1_filter_tile - Point<DIM>(centers);
        const Point<DIM> l2_input_start = subrect.lo - Point<DIM>(centers);
        const Point<DIM> l2_input_stop  = subrect.lo + l2_output_tile - one + Point<DIM>(extents) -
                                         filter_rect.lo - one - Point<DIM>(centers);
        convolution_large_tile<VAL, DIM, THREADVALS>
          <<<properties.multiProcessorCount, CONVOLUTION_THREADS, dynamic_smem>>>(out,
                                                                                  filter,
                                                                                  in,
                                                                                  root_rect,
                                                                                  subrect,
                                                                                  filter_rect,
                                                                                  l2_input_start,
                                                                                  l2_input_stop,
                                                                                  l1_input_start,
                                                                                  zero,
                                                                                  one,
                                                                                  args);
      }
    }
  }
};

template <typename VAL, int DIM>
__host__ static inline size_t create_forward_plan(cufftHandle plan, const Point<DIM>& size)
{
  // should always call specialized templates
  assert(false);
  return 0;
}

template <>
__host__ inline size_t create_forward_plan<float, 1>(cufftHandle plan, const Point<1>& size)
{
  size_t workarea = 0;
  CHECK_CUFFT(cufftMakePlan1d(plan, size[0], CUFFT_R2C, 1 /*batch*/, &workarea));
  return workarea;
}

template <>
__host__ inline size_t create_forward_plan<double, 1>(cufftHandle plan, const Point<1>& size)
{
  size_t workarea = 0;
  CHECK_CUFFT(cufftMakePlan1d(plan, size[0], CUFFT_D2Z, 1 /*batch*/, &workarea));
  return workarea;
}

template <>
__host__ inline size_t create_forward_plan<float, 2>(cufftHandle plan, const Point<2>& size)
{
  size_t workarea = 0;
  CHECK_CUFFT(cufftMakePlan2d(plan, size[0], size[1], CUFFT_R2C, &workarea));
  return workarea;
}

template <>
__host__ inline size_t create_forward_plan<double, 2>(cufftHandle plan, const Point<2>& size)
{
  size_t workarea = 0;
  CHECK_CUFFT(cufftMakePlan2d(plan, size[0], size[1], CUFFT_D2Z, &workarea));
  return workarea;
}

template <>
__host__ inline size_t create_forward_plan<float, 3>(cufftHandle plan, const Point<3>& size)
{
  size_t workarea = 0;
  CHECK_CUFFT(cufftMakePlan3d(plan, size[0], size[1], size[2], CUFFT_R2C, &workarea));
  return workarea;
}

template <>
__host__ inline size_t create_forward_plan<double, 3>(cufftHandle plan, const Point<3>& size)
{
  size_t workarea = 0;
  CHECK_CUFFT(cufftMakePlan3d(plan, size[0], size[1], size[2], CUFFT_D2Z, &workarea));
  return workarea;
}

template <typename VAL, int DIM>
__host__ static inline size_t create_backward_plan(cufftHandle plan, const Point<DIM>& size)
{
  // should always call specialized templates
  assert(false);
  return 0;
}

template <>
__host__ inline size_t create_backward_plan<float, 1>(cufftHandle plan, const Point<1>& size)
{
  size_t workarea = 0;
  CHECK_CUFFT(cufftMakePlan1d(plan, size[0], CUFFT_C2R, 1 /*batch*/, &workarea));
  return workarea;
}

template <>
__host__ inline size_t create_backward_plan<double, 1>(cufftHandle plan, const Point<1>& size)
{
  size_t workarea = 0;
  CHECK_CUFFT(cufftMakePlan1d(plan, size[0], CUFFT_Z2D, 1 /*batch*/, &workarea));
  return workarea;
}

template <>
__host__ inline size_t create_backward_plan<float, 2>(cufftHandle plan, const Point<2>& size)
{
  size_t workarea = 0;
  CHECK_CUFFT(cufftMakePlan2d(plan, size[0], size[1], CUFFT_C2R, &workarea));
  return workarea;
}

template <>
__host__ inline size_t create_backward_plan<double, 2>(cufftHandle plan, const Point<2>& size)
{
  size_t workarea = 0;
  CHECK_CUFFT(cufftMakePlan2d(plan, size[0], size[1], CUFFT_Z2D, &workarea));
  return workarea;
}

template <>
__host__ inline size_t create_backward_plan<float, 3>(cufftHandle plan, const Point<3>& size)
{
  size_t workarea = 0;
  CHECK_CUFFT(cufftMakePlan3d(plan, size[0], size[1], size[2], CUFFT_C2R, &workarea));
  return workarea;
}

template <>
__host__ inline size_t create_backward_plan<double, 3>(cufftHandle plan, const Point<3>& size)
{
  size_t workarea = 0;
  CHECK_CUFFT(cufftMakePlan3d(plan, size[0], size[1], size[2], CUFFT_Z2D, &workarea));
  return workarea;
}

template <typename VAL>
__host__ static inline void cufft_execute_forward(cufftHandle plan, VAL* idata, VAL* odata)
{
  assert(false);  // should never be called
}

template <>
__host__ inline void cufft_execute_forward<float>(cufftHandle plan, float* idata, float* odata)
{
  CHECK_CUFFT(cufftExecR2C(plan, (cufftReal*)idata, (cufftComplex*)odata));
}

template <>
__host__ inline void cufft_execute_forward<double>(cufftHandle plan, double* idata, double* odata)
{
  CHECK_CUFFT(cufftExecD2Z(plan, (cufftDoubleReal*)idata, (cufftDoubleComplex*)odata));
}

template <typename VAL>
__host__ static inline void cufft_execute_backward(cufftHandle plan, VAL* idata, VAL* odata)
{
  assert(false);  // should never be called
}

template <>
__host__ inline void cufft_execute_backward<float>(cufftHandle plan, float* idata, float* odata)
{
  CHECK_CUFFT(cufftExecC2R(plan, (cufftComplex*)idata, (cufftReal*)odata));
}

template <>
__host__ inline void cufft_execute_backward<double>(cufftHandle plan, double* idata, double* odata)
{
  CHECK_CUFFT(cufftExecZ2D(plan, (cufftDoubleComplex*)idata, (cufftDoubleReal*)odata));
}

template<typename VAL>
__host__ static inline void cufft_set_callback(cufftHandle plan, void *callback, void *data,
                                               bool forward, bool load)
{
  assert(false); // should never be called
}

template<>
__host__ inline void cufft_set_callback<float>(cufftHandle plan, void *callback, void *data,
                                               bool forward, bool load)
{
  void *callbacks[1] = { callback };
  void *datas[1] = { data };
  CHECK_CUFFT( cufftXtSetCallback(plan, callbacks, 
        forward ? (load ? CUFFT_CB_LD_REAL : CUFFT_CB_ST_COMPLEX) :
                  (load ? CUFFT_CB_LD_COMPLEX : CUFFT_CB_ST_REAL), datas) );
}

template<>
__host__ inline void cufft_set_callback<double>(cufftHandle plan, void *callback, void *data,
                                                bool forward, bool load)
{
  void *callbacks[1] = { callback };
  void *datas[1] = { data };
  CHECK_CUFFT( cufftXtSetCallback(plan, callbacks,
        forward ? (load ? CUFFT_CB_LD_REAL_DOUBLE : CUFFT_CB_ST_COMPLEX_DOUBLE) :
                  (load ? CUFFT_CB_LD_COMPLEX_DOUBLE : CUFFT_CB_ST_REAL_DOUBLE), datas) );
}

struct ZeroPadLoadData {
  FastDivmodU64 pitches[3];
  size_t strides[3];
  size_t bounds[3];
  int dim;
};

template<typename T>
__device__ T load_zero_pad(void *data, size_t offset,
                           void *callerinfo,
                           void *sharedptr)
{
  const ZeroPadLoadData *info = (const ZeroPadLoadData*)callerinfo;
  size_t actual = 0;
  #pragma unroll 3
  for (int d = 0; d < info->dim; d++) {
    coord_t coord = info->pitches[d].divmod(offset, offset);
    if (coord >= info->bounds[d])
      return 0.f;
    actual += coord * info->strides[d];
  }
  T *ptr = (T*)data;
  return ptr[actual];
}

__device__ cufftCallbackLoadR d_load_float_zero_pad = load_zero_pad<cufftReal>;
__device__ cufftCallbackLoadD d_load_double_zero_pad = load_zero_pad<cufftDoubleReal>;

template<typename VAL>
__host__ static inline void* load_zero_pad_callback(void)
{
  assert(false); // should never be called 
  return nullptr;
}

template<>
__host__ inline void* load_zero_pad_callback<float>(void)
{
  void *ptr = nullptr;
  CHECK_CUDA( cudaMemcpyFromSymbol(&ptr, d_load_float_zero_pad, sizeof(ptr)) );
  assert(ptr != nullptr);
  return ptr;
}

template<>
__host__ inline void* load_zero_pad_callback<double>(void)
{
  void *ptr = nullptr;
  CHECK_CUDA( cudaMemcpyFromSymbol(&ptr, d_load_double_zero_pad, sizeof(ptr)) );
  assert(ptr != nullptr);
  return ptr;
}

struct LoadComplexData {
  size_t buffervolume;
};


__device__ cufftComplex load_complex_float_multiply(void *data, size_t offset,
                                                    void *callerinfo,
                                                    void *sharedptr)
{
  const LoadComplexData *info = (const LoadComplexData*)callerinfo;
  cufftComplex *ptr = (cufftComplex*)data;
  cufftComplex lhs = ptr[offset];
  cufftComplex rhs = ptr[offset+info->buffervolume];
  return make_cuComplex(lhs.x * rhs.x - lhs.y * rhs.y, lhs.x * rhs.y + lhs.y * rhs.x);
}

__device__ cufftCallbackLoadC d_complex_float_multiply = load_complex_float_multiply;

__device__ cufftDoubleComplex load_complex_double_multiply(void *data, size_t offset,
                                                           void *callerinfo,
                                                           void *sharedptr)
{
  const LoadComplexData *info = (const LoadComplexData*)callerinfo;
  cufftDoubleComplex *ptr = (cufftDoubleComplex*)data;
  cufftDoubleComplex lhs = ptr[offset];
  cufftDoubleComplex rhs = ptr[offset+info->buffervolume];
  return make_cuDoubleComplex(lhs.x * rhs.x - lhs.y * rhs.y, lhs.x * rhs.y + lhs.y * rhs.x);
}

__device__ cufftCallbackLoadZ d_complex_double_multiply = load_complex_double_multiply;

template<typename VAL>
__host__ static inline void* load_multiply_callback(void)
{
  assert(false); // should never be called 
  return nullptr;
}

template<>
__host__ inline void* load_multiply_callback<float>(void)
{
  void *ptr = nullptr;
  CHECK_CUDA( cudaMemcpyFromSymbol(&ptr, d_complex_float_multiply, sizeof(ptr)) );
  assert(ptr != nullptr);
  return ptr;
}

template<>
__host__ inline void* load_multiply_callback<double>(void)
{
  void *ptr = nullptr;
  CHECK_CUDA( cudaMemcpyFromSymbol(&ptr, d_complex_double_multiply, sizeof(ptr)) );
  assert(ptr != nullptr);
  return ptr;
}

template<typename T>
struct StoreOutputData {
  FastDivmodU64 pitches[3];
  size_t offsets[3];
  size_t strides[3];
  size_t bounds[3];
  T scale_factor;
  int dim;
};

template<typename T>
__device__ void store_output(void *data, size_t offset,
                             T value,
                             void *callerinfo,
                             void *sharedptr)
{
  const StoreOutputData<T> *info = (const StoreOutputData<T>*)callerinfo;
  size_t actual = 0;
  #pragma unroll 3
  for (int d = 0; d < info->dim; d++) {
    coord_t coord = info->pitches[d].divmod(offset, offset);
    coord -= info->offsets[d];
    if (coord < 0)
      return;
    if (coord >= info->bounds[d])
      return;
    actual += coord * info->strides[d];
  }
  T *ptr = (T*)data;
  ptr[actual] = value * info->scale_factor;
}

__device__ cufftCallbackStoreR d_store_float_output = store_output<cufftReal>;
__device__ cufftCallbackStoreD d_store_double_output = store_output<cufftDoubleReal>;

template<typename VAL>
__host__ static inline void* load_store_callback(void)
{
  assert(false); // should never be called 
  return nullptr;
}

template<>
__host__ inline void* load_store_callback<float>(void)
{
  void *ptr = nullptr;
  CHECK_CUDA( cudaMemcpyFromSymbol(&ptr, d_store_float_output, sizeof(ptr)) );
  assert(ptr != nullptr);
  return ptr;
}

template<>
__host__ inline void* load_store_callback<double>(void)
{
  void *ptr = nullptr;
  CHECK_CUDA( cudaMemcpyFromSymbol(&ptr, d_store_double_output, sizeof(ptr)) );
  assert(ptr != nullptr);
  return ptr;
}

template <typename VAL, int DIM>
__host__ static inline void cufft_convolution(AccessorWO<VAL, DIM> out,
                                              AccessorRO<VAL, DIM> filter,
                                              AccessorRO<VAL, DIM> in,
                                              const Rect<DIM>& root_rect,
                                              const Rect<DIM>& subrect,
                                              const Rect<DIM>& filter_rect)
{
  int device;
  CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp properties;
  CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
  size_t max_smem_size = properties.sharedMemPerBlockOptin;

  // Only need to do these calls the first time on each device so
  // we use a bit mask to track which devices we've done it for
  static unsigned long long mask = 0;
  if (!(mask & (1 << device))) {
    if (properties.sharedMemPerBlock < max_smem_size) {
      CHECK_CUDA(cudaFuncSetAttribute(convolution_small_tile1<VAL, DIM>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      max_smem_size));
      CHECK_CUDA(cudaFuncSetAttribute(convolution_small_tile2<VAL, DIM>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      max_smem_size));
    }
    if (sizeof(VAL) >= 8) {
      // Only need to set this on the first invocation
      CHECK_CUDA(cudaFuncSetSharedMemConfig(convolution_small_tile1<VAL, DIM>,
                                            cudaSharedMemBankSizeEightByte));
      CHECK_CUDA(cudaFuncSetSharedMemConfig(convolution_small_tile2<VAL, DIM>,
                                            cudaSharedMemBankSizeEightByte));
    }
    // Make sure we have enough bits for every device
    assert(device < (8 * sizeof(mask)));
    // Make sure not to race with updates from other GPUs
    __sync_fetch_and_add(&mask, (1 << device));
  }
  unsigned extents[DIM];
  unsigned centers[DIM];
  for (int d = 0; d < DIM; d++) {
    assert(filter_rect.lo[d] == 0);
    extents[d] = filter_rect.hi[d] + 1;
    centers[d] = static_cast<coord_t>(extents[d] / 2);
  }
  Point<DIM> tile;
  for (int d = DIM - 1; d >= 0; d--) {
    // Make sure that each tile is at least double the size of the filter
    // so that we can get some savings in bandwidth needed
    tile[d] = 2 * centers[d];
    if (d == (DIM - 1)) {
      // In order to maximize bandwidth, we want to make sure we're loading at
      // least 128B of contiguous memory along the last axis (row-major) of input
      const unsigned min_contig_elmts = 128 / sizeof(VAL);
      if ((tile[d] + 2 * centers[d]) < min_contig_elmts)
        tile[d] = min_contig_elmts - 2 * centers[d];
    }
  }
  unsigned smem_size = sizeof(VAL);
  for (int d = 0; d < DIM; d++) smem_size *= (tile[d] + 2 * centers[d]);
  if (smem_size <= max_smem_size) {
    launch_small_tile_kernel<VAL, DIM>(out,
                                       filter,
                                       in,
                                       root_rect,
                                       subrect,
                                       filter_rect,
                                       properties,
                                       extents,
                                       centers,
                                       tile,
                                       smem_size,
                                       max_smem_size);
  } else {
    // cufft plans are awful things that call cudaMalloc/cudaFree which
    // completely destroys asynchronous execution so we need to cache
    // these plans to avoid calling it as often as possible
    constexpr size_t MAX_PLANS = 4; 
    struct cufftPlan {
    public:
      cufftPlan(void) : fftshape(Point<DIM>::ZEROES()) { }
    public:
      cufftHandle forward;
      cufftHandle backward;
      Point<DIM> fftshape;
      void *load_callback; // for loading zero-pad inputs
      size_t workarea_size; 
      unsigned lru_index;
      ZeroPadLoadData *filter_load_data_d;
      ZeroPadLoadData filter_load_data;
      ZeroPadLoadData *signal_load_data_d;
      ZeroPadLoadData signal_load_data;
      LoadComplexData *complex_load_data_d;
      LoadComplexData complex_load_data;
      StoreOutputData<VAL> *store_output_data_d;
      StoreOutputData<VAL> store_output_data;
    };
    static cufftPlan cufft_plan_cache[LEGION_MAX_NUM_PROCS][MAX_PLANS];
    // Instead of doing the large tile case, we can instead do this
    // by transforming both the input and the filter to the frequency
    // domain using an FFT, perform the convolution with a point-wise
    // multiplication, and then transform the result back to the spatial domain
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // First compute how big our temporary allocation needs to be
    // We'll need two of them to store the zero-padded data for the inputs
    const Point<DIM> zero = Point<DIM>::ZEROES();
    const Point<DIM> one  = Point<DIM>::ONES();
    Rect<DIM> offset_bounds;
    for (int d = 0; d < DIM; d++) {
      offset_bounds.lo[d] = subrect.lo[d] - centers[d];
      offset_bounds.hi[d] = subrect.hi[d] + extents[d] - 1 - centers[d];
    }
    Rect<DIM> input_bounds         = root_rect.intersection(offset_bounds);
    const Point<DIM> signal_bounds = input_bounds.hi - input_bounds.lo + one;
    const Point<DIM> filter_bounds = filter_rect.hi - filter_rect.lo + one;
    Point<DIM> fftsize = signal_bounds + filter_bounds;
    for (int d = 0; d < DIM; d++) {
      if ((fftsize[d] % 2) == 1)
        fftsize[d] -= 1;
    }
    // Cufft needs the last dimension to have fftsize/2+1 complex elements for
    // the temporary buffer, we know that the last dimension is already even
    // so we just need to add two elements to the last dim of the fftsize
    Point<DIM> buffersize = fftsize;
    buffersize[DIM - 1] = fftsize[DIM - 1] + 2;
    size_t buffervolume = 1;
    for (int d = 0; d < DIM; d++) buffervolume *= buffersize[d];
    // In theory we could do this with a single output buffer by doing
    // += operations in the second forward FFT kernel into the buffer,
    // but unfortunately cufft likes to use the output buffer during its
    // execution and that destroys the data from the first FFT
    DeferredBuffer<VAL,1> buffer(Rect<1>(Point<1>(0), Point<1>(2*buffervolume-1)),
                          Memory::GPU_FB_MEM, nullptr /*initial*/, 128 /*alignment*/);
    VAL* buffer_ptr = buffer.ptr(Point<1>(0));
    // Check to see if the plan is already in the cache
    int plan_index = -1;
    // Some hackiness until Legion can support stateless runtime caches 
    const unsigned proc_idx = 
      Processor::get_executing_processor().id & (LEGION_MAX_NUM_PROCS - 1);
    for (unsigned idx = 0; idx < MAX_PLANS; idx++) {
      if (fftsize != cufft_plan_cache[proc_idx][idx].fftshape)
        continue;
      plan_index = idx;
      break;
    }
    if (plan_index < 0) {
      // If we didn't find it, then we'll need to allocate an index for it,
      // see if there are any unused ones, otherwise fine the one that was
      // least recently-used
      for (unsigned idx = 0; idx < MAX_PLANS; idx++) {
        if (zero == cufft_plan_cache[proc_idx][idx].fftshape) {
          // Previously uninitialized plan so we can use it
          plan_index = idx;
          // Set the lru_index
          cufftPlan &plan = cufft_plan_cache[proc_idx][idx];
          plan.lru_index = idx;
          plan.load_callback = load_zero_pad_callback<VAL>();
          CHECK_CUDA(cudaMalloc(&plan.filter_load_data_d, sizeof(ZeroPadLoadData)));
          plan.filter_load_data.dim = 0;
          CHECK_CUDA(cudaMalloc(&plan.signal_load_data_d, sizeof(ZeroPadLoadData)));
          plan.signal_load_data.dim = 0;
          CHECK_CUDA(cudaMalloc(&plan.complex_load_data_d, sizeof(LoadComplexData)));
          plan.complex_load_data.buffervolume = 0;
          CHECK_CUDA(cudaMalloc(&plan.store_output_data_d, sizeof(StoreOutputData<VAL>)));
          plan.store_output_data.dim = 0;
          break;
        } else if (cufft_plan_cache[proc_idx][idx].lru_index == (MAX_PLANS-1)) {
          // Destroy the resources associated with the previous plan
          cufftPlan &plan = cufft_plan_cache[proc_idx][idx];
          CHECK_CUFFT(cufftDestroy(plan.forward));
          CHECK_CUFFT(cufftDestroy(plan.backward));
          plan_index = idx;
          break;
        }
      }
      assert(plan_index >= 0); 
      // Create the plans for going in both directions
      cufftPlan &plan = cufft_plan_cache[proc_idx][plan_index];
      plan.fftshape = fftsize;
      CHECK_CUFFT(cufftCreate(&plan.forward));
      CHECK_CUFFT(cufftSetAutoAllocation(plan.forward, 0 /*we'll do the allocation*/));
      CHECK_CUFFT(cufftSetStream(plan.forward, stream));
      plan.workarea_size = create_forward_plan<VAL, DIM>(plan.forward, fftsize); 
      CHECK_CUFFT(cufftCreate(&plan.backward));
      CHECK_CUFFT(cufftSetAutoAllocation(plan.backward, 0 /*we'll do the allocation*/));
      size_t backward_size = create_backward_plan<VAL, DIM>(plan.backward, fftsize);
      if (plan.workarea_size < backward_size)
        plan.workarea_size = backward_size;
      cufft_set_callback<VAL>(plan.backward, load_multiply_callback<VAL>(),
          plan.complex_load_data_d, false/*forward*/, true/*load*/);
      cufft_set_callback<VAL>(plan.backward, load_store_callback<VAL>(),
          plan.store_output_data_d, false/*forward*/, false/*load*/);
    }
    assert(plan_index >= 0); 
    cufftPlan &plan = cufft_plan_cache[proc_idx][plan_index];
    // Set the stream and working area for the plans
    CHECK_CUFFT(cufftSetStream(plan.forward, stream));
    CHECK_CUFFT(cufftSetStream(plan.backward, stream));
    // Allocate a temporary buffer for it if it needs one
    DeferredBuffer<uint8_t, 1> workarea_buffer;
    if (plan.workarea_size > 0) {
      const Point<1> zero1d(0);
      workarea_buffer = DeferredBuffer<uint8_t, 1>(Rect<1>(zero1d,
                                                   Point<1>(plan.workarea_size - 1)),
                                                   Memory::GPU_FB_MEM,
                                                   nullptr /*initial*/,
                                                   128 /*alignment*/);
      void* workarea  = workarea_buffer.ptr(zero1d);
      CHECK_CUFFT(cufftSetWorkArea(plan.forward, workarea));
      CHECK_CUFFT(cufftSetWorkArea(plan.backward, workarea));
    }
    // FFT the filter data in-place in buffer
    // Set the load callback
    size_t strides[DIM];
    const VAL *filter_ptr = filter.ptr(filter_rect, strides);
    ZeroPadLoadData &filter_data = plan.filter_load_data;
    bool need_copy = filter_data.dim != DIM;
    if (need_copy)
      filter_data.dim = DIM;
    size_t pitch = 1;
    for (int d = DIM-1; d >= 0; d--) {
      if (need_copy || (filter_data.pitches[d].divisor != pitch)) {
        filter_data.pitches[d] = FastDivmodU64(pitch);
        need_copy = true;
      }
      pitch *= fftsize[d];
      if (need_copy || (filter_data.strides[d] != strides[d])) {
        filter_data.strides[d] = strides[d];
        need_copy = true;
      }
      if (need_copy || (filter_data.bounds[d] != filter_bounds[d])) {
        filter_data.bounds[d] = filter_bounds[d];
        need_copy = true;
      }
    }
    if (need_copy)
      CHECK_CUDA(cudaMemcpyAsync(plan.filter_load_data_d, &filter_data,
            sizeof(ZeroPadLoadData), cudaMemcpyHostToDevice, stream));
    cufft_set_callback<VAL>(plan.forward, plan.load_callback, plan.filter_load_data_d,
                            true/*forward*/, true/*load*/);
    cufft_execute_forward<VAL>(plan.forward, const_cast<VAL*>(filter_ptr), buffer_ptr);
    // FFT the input data from the input into the buffer
    // Use a cufft callback to perform point-wise multiplication on the way out
    // If the signal is dense in memory we don't need a load callback
    const VAL *signal_ptr = in.ptr(input_bounds, strides);
    ZeroPadLoadData &signal_data = plan.signal_load_data;
    need_copy = signal_data.dim != DIM;
    if (need_copy)
      signal_data.dim = DIM;
    pitch = 1;
    for (int d = DIM-1; d >= 0; d--) {
      if (need_copy || (signal_data.pitches[d].divisor != pitch)) {
        signal_data.pitches[d] = FastDivmodU64(pitch);
        need_copy = true;
      }
      pitch *= fftsize[d];
      if (need_copy || (signal_data.strides[d] != strides[d])) {
        signal_data.strides[d] = strides[d];
        need_copy = true;
      }
      if (need_copy || (signal_data.bounds[d] != signal_bounds[d])) {
        signal_data.bounds[d] = signal_bounds[d];
        need_copy = true;
      } 
    }
    if (need_copy)
      CHECK_CUDA(cudaMemcpyAsync(plan.signal_load_data_d, &signal_data, 
              sizeof(ZeroPadLoadData), cudaMemcpyHostToDevice, stream));
    cufft_set_callback<VAL>(plan.forward, plan.load_callback, plan.signal_load_data_d,
                            true/*forward*/, true/*load*/);
    cufft_execute_forward<VAL>(plan.forward, const_cast<VAL*>(signal_ptr),
                               buffer_ptr+buffervolume/*second half of the buffer*/);
    // Inverse FFT for the output in-place in the temporary buffer
    if (plan.complex_load_data.buffervolume != (buffervolume/2)) {
      plan.complex_load_data.buffervolume = buffervolume/2;
      CHECK_CUDA(cudaMemcpyAsync(plan.complex_load_data_d, &plan.complex_load_data,
            sizeof(LoadComplexData), cudaMemcpyHostToDevice, stream));
    }
    VAL *output_ptr = out.ptr(subrect, strides);
    StoreOutputData<VAL> &store_data = plan.store_output_data;
    need_copy = store_data.dim != DIM;
    if (need_copy)
      store_data.dim = DIM;
    Point<DIM> buffer_offset;
    for (int d = 0; d < DIM; d++)
      buffer_offset[d] =
        centers[d] - (((extents[d] % 2) == 0) ? 1 : 0) +
        ((offset_bounds.lo[d] < root_rect.lo[d]) ? (subrect.lo[d] - root_rect.lo[d]) : centers[d]);
    pitch = 1;
    Point<DIM> output_bounds = subrect.hi - subrect.lo + one;
    for (int d = DIM-1; d >= 0; d--) {
      if (need_copy || (store_data.pitches[d].divisor != pitch)) {
        store_data.pitches[d] = FastDivmodU64(pitch);
        need_copy = true;
      }
      pitch *= fftsize[d];
      if (need_copy || (store_data.offsets[d] != buffer_offset[d])) {
        store_data.offsets[d] = buffer_offset[d];
        need_copy = true;
      }
      if (need_copy || (store_data.strides[d] != strides[d])) {
        store_data.strides[d] = strides[d];
        need_copy = true;
      }
      if (need_copy || (store_data.bounds[d] != output_bounds[d])) {
        store_data.bounds[d] = output_bounds[d];
        need_copy = true;
      }
    }
    if (need_copy) {
      // If all the pitches matched then we know the scale factor is the same
      store_data.scale_factor = 1.0/pitch;
      CHECK_CUDA(cudaMemcpyAsync(plan.store_output_data_d, &store_data,
            sizeof(StoreOutputData<VAL>), cudaMemcpyHostToDevice, stream));
    }
    cufft_execute_backward<VAL>(plan.backward, buffer_ptr, output_ptr); 
#if 0
    // This is useful debugging code for finding the output
    VAL *debug_buffer = (VAL*)malloc(buffervolume*sizeof(VAL));
    CHECK_CUDA( cudaMemcpyAsync(debug_buffer, buffer_ptr+buffervolume, buffervolume*sizeof(VAL), cudaMemcpyDeviceToHost, stream) );
    CHECK_CUDA( cudaStreamSynchronize(stream) );
    for (unsigned idx = 0; idx < buffervolume; idx++) {
      if ((idx % fftsize[DIM-1]) == 0)
        printf("\n");
      printf("%.8g ", debug_buffer[idx]*scaling_factor);
    }
    printf("\n");
    free(debug_buffer);
#endif
    // Bump the lru_index of any plans that were less than our lru_index
    // and then set our lru_index back to zero
    if (plan.lru_index > 0) {
      for (unsigned idx = 0; idx < MAX_PLANS; idx++) {
        cufftPlan &other = cufft_plan_cache[proc_idx][idx];
        if (other.lru_index < plan.lru_index)
          other.lru_index++;
      }
      plan.lru_index = 0;
    }
    // Clean up our resources, DeferredBuffers are cleaned up by Legion
    CHECK_CUDA( cudaStreamDestroy(stream) );
  }
}

// Below here are specializations of the template for dimensions 1,2,3
// and fp32 and fp64 so we can use cufft to do the fft
template <>
struct ConvolveImplBody<VariantKind::GPU, FLOAT_LT, 1> {
  using VAL = legate_type_of<FLOAT_LT>;

  __host__ void operator()(AccessorWO<VAL, 1> out,
                           AccessorRO<VAL, 1> filter,
                           AccessorRO<VAL, 1> in,
                           const Rect<1>& root_rect,
                           const Rect<1>& subrect,
                           const Rect<1>& filter_rect) const
  {
    cufft_convolution<VAL, 1>(out, filter, in, root_rect, subrect, filter_rect);
  }
};

template <>
struct ConvolveImplBody<VariantKind::GPU, FLOAT_LT, 2> {
  using VAL = legate_type_of<FLOAT_LT>;

  __host__ void operator()(AccessorWO<VAL, 2> out,
                           AccessorRO<VAL, 2> filter,
                           AccessorRO<VAL, 2> in,
                           const Rect<2>& root_rect,
                           const Rect<2>& subrect,
                           const Rect<2>& filter_rect) const
  {
    cufft_convolution<VAL, 2>(out, filter, in, root_rect, subrect, filter_rect);
  }
};

template <>
struct ConvolveImplBody<VariantKind::GPU, FLOAT_LT, 3> {
  using VAL = legate_type_of<FLOAT_LT>;

  __host__ void operator()(AccessorWO<VAL, 3> out,
                           AccessorRO<VAL, 3> filter,
                           AccessorRO<VAL, 3> in,
                           const Rect<3>& root_rect,
                           const Rect<3>& subrect,
                           const Rect<3>& filter_rect) const
  {
    cufft_convolution<VAL, 3>(out, filter, in, root_rect, subrect, filter_rect);
  }
};

template <>
struct ConvolveImplBody<VariantKind::GPU, DOUBLE_LT, 1> {
  using VAL = legate_type_of<DOUBLE_LT>;

  __host__ void operator()(AccessorWO<VAL, 1> out,
                           AccessorRO<VAL, 1> filter,
                           AccessorRO<VAL, 1> in,
                           const Rect<1>& root_rect,
                           const Rect<1>& subrect,
                           const Rect<1>& filter_rect) const
  {
    cufft_convolution<VAL, 1>(out, filter, in, root_rect, subrect, filter_rect);
  }
};

template <>
struct ConvolveImplBody<VariantKind::GPU, DOUBLE_LT, 2> {
  using VAL = legate_type_of<DOUBLE_LT>;

  __host__ void operator()(AccessorWO<VAL, 2> out,
                           AccessorRO<VAL, 2> filter,
                           AccessorRO<VAL, 2> in,
                           const Rect<2>& root_rect,
                           const Rect<2>& subrect,
                           const Rect<2>& filter_rect) const
  {
    cufft_convolution<VAL, 2>(out, filter, in, root_rect, subrect, filter_rect);
  }
};

template <>
struct ConvolveImplBody<VariantKind::GPU, DOUBLE_LT, 3> {
  using VAL = legate_type_of<DOUBLE_LT>;

  __host__ void operator()(AccessorWO<VAL, 3> out,
                           AccessorRO<VAL, 3> filter,
                           AccessorRO<VAL, 3> in,
                           const Rect<3>& root_rect,
                           const Rect<3>& subrect,
                           const Rect<3>& filter_rect) const
  {
    cufft_convolution<VAL, 3>(out, filter, in, root_rect, subrect, filter_rect);
  }
};

/*static*/ void ConvolveTask::gpu_variant(TaskContext& context)
{
  convolve_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
