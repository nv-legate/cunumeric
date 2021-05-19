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

#include <iostream>

#include "unary/unary_red.h"
#include "unary/unary_red_util.h"
#include "core.h"
#include "deserializer.h"
#include "dispatch.h"
#include "point_task.h"

namespace legate {
namespace numpy {

using namespace Legion;

namespace gpu {

template <typename T>
constexpr T div_and_ceil(T value, T divider)
{
  return std::max<T>((value + divider - 1) / divider, 1);
}

static constexpr coord_t WARP_SIZE = 32;

// This helper class is to compute the shape of thread blocks for reduction kernels.
// The strategy is to parallelize on dimensions, from the outermost one to the innermost,
// that are not being collapsed, thereby having threads work on independet lanes of
// reductions as much as possible. In case where the non-collapsing dimensions don't
// have enough elements to be assigned to the threads, we also parallelize on
// the collapsing domain. One exceptional case to this strategy is where the collapsing
// dimension is the innermost one, in which case we prefer that dimension to the others
// in order to enjoy wrap coalescing. The maximum degree of such parallelism woudl be 32,
// which is the size of a wrap.
template <int32_t DIM>
struct ThreadBlock {
  void initialize(const Rect<DIM> &domain, int32_t collapsed_dim)
  {
    auto remaining = static_cast<coord_t>(THREADS_PER_BLOCK);

    Point<DIM> domain_extents;
    for (int32_t idx = 0; idx < DIM; ++idx)
      domain_extents[idx] = domain.hi[idx] - domain.lo[idx] + 1;

    // If the innermost dimension is being collapsed, we assign at least one warp to it
    // for warp coalsecing.
    if (collapsed_dim == DIM - 1) {
      auto extent             = std::min<coord_t>(WARP_SIZE, domain_extents[collapsed_dim]);
      extents_[collapsed_dim] = extent;
      remaining               = std::max<coord_t>(remaining / extent, 1);
    }

    // Then, we compute how many threads there should be along aech dimension,
    // excluding the one being collapsed
    for (int32_t idx = DIM - 1; idx >= 0; --idx) {
      if (idx == collapsed_dim) continue;
      auto extent   = std::min(remaining, domain_extents[idx]);
      extents_[idx] = extent;
      remaining     = std::max<coord_t>(remaining / extent, 1);
    }

    // Finally, we determine degree of parallelism for the collapsed dimension if we didn't above
    if (collapsed_dim != DIM - 1)
      extents_[collapsed_dim] = std::min(remaining, domain_extents[collapsed_dim]);

    // Cache the aggregate number of threads per increment in each dimension,
    // which later will be used for de-linearization of a thread id
    num_threads_ = 1;
    for (int32_t idx = DIM - 1; idx >= 0; --idx) {
      pitches_[idx] = num_threads_;
      num_threads_ *= extents_[idx];
    }
  }

  // Compute a relative coordiate of a given thread
  __host__ __device__ Point<DIM> point(coord_t tid) const
  {
    Point<DIM> p;
    for (int32_t dim = 0; dim < DIM; ++dim) {
      p[dim] = tid / pitches_[dim];
      tid    = tid % pitches_[dim];
    }
    return p;
  }

  // Total number of threads
  size_t num_threads_;
  // Number of threads along each dimension
  Point<DIM> extents_;
  // Aggregate number of threads per increment in each dimension
  Point<DIM> pitches_;
};

// This class represents a set of concurrent thread blocks. Concurrent thread blocks form
// hyperplanes in N-dimensional integer lattice such that the collapsed dimension is normal to them.
// The size of thread blocks is determined by the maximum number of CTAs for a given kernel;
// the number of concurrent thread blocks is the minimum number of hyperplanes whose aggregate
// volume exceeds the maximum number of CTAs.
template <int32_t DIM>
struct ThreadBlocks {
  void initialize(const Rect<DIM> &domain, int32_t collapsed_dim)
  {
    collapsed_dim_ = collapsed_dim;
    block_.initialize(domain, collapsed_dim);

    for (int32_t idx = 0; idx < DIM; ++idx) {
      auto domain_extent = domain.hi[idx] - domain.lo[idx] + 1;
      extents_[idx]      = div_and_ceil(domain_extent, block_.extents_[idx]);
    }

    // We want the collapsed dimension to be the outermost one when
    // de-linearizing the block id.
    dim_order_[0] = collapsed_dim_;
    for (int32_t dim = 0, idx = 1; dim < DIM; ++dim)
      if (dim != collapsed_dim_) dim_order_[idx++] = dim;

    // Compute the aggregate number of blocks per increment in each dimension
    coord_t num_blocks = 1;
    for (int32_t idx = DIM - 1; idx >= 0; --idx) {
      auto dim      = dim_order_[idx];
      pitches_[dim] = num_blocks;
      num_blocks *= extents_[dim];
    }
    // For now we say all blocks can run concurrent.
    num_blocks_ = num_blocks;
    // Also compute the stride on the collapsed dimension
    collapsed_dim_stride_ = extents_[collapsed_dim_] * block_.extents_[collapsed_dim_];
  }

  // De-linearized the linearized block id and thread it into an N-dimensional point
  __host__ __device__ Point<DIM> point(coord_t bid, coord_t tid, const Point<DIM> &origin) const
  {
    Point<DIM> p = origin;
    for (int32_t dim : dim_order_) {
      p[dim] += (bid / pitches_[dim]) * block_.extents_[dim];
      bid = bid % pitches_[dim];
    }
    p += block_.point(tid);
    return p;
  }

  void compute_maximum_concurrency(const void *func)
  {
    int32_t num_ctas = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_ctas, func, num_threads(), 0);

    size_t plane_size = pitches_[collapsed_dim_];
    // Calculate the number of planes whose volume barely exceeds the maximum number of CTAs
    size_t max_num_concurrent_planes =
      std::max<size_t>(div_and_ceil<size_t>(num_ctas, plane_size), 1);
    // Then we update the number of concurrent thread blocks and the stride on the collapsed
    // dimension
    num_blocks_           = plane_size * max_num_concurrent_planes;
    collapsed_dim_stride_ = max_num_concurrent_planes * block_.extents_[collapsed_dim_];
  }

  __host__ __device__ inline void next_point(Point<DIM> &point) const
  {
    point[collapsed_dim_] += collapsed_dim_stride_;
  }

  constexpr size_t num_blocks() const { return num_blocks_; }
  constexpr size_t num_threads() const { return block_.num_threads_; }

  // List of dimensions, from the outermost one to the innermost
  int32_t dim_order_[DIM];
  int32_t collapsed_dim_;
  coord_t collapsed_dim_stride_;
  // Shape of each thread block
  ThreadBlock<DIM> block_;
  // Number of thread blocks along each dimension
  Point<DIM> extents_;
  // Aggregate number of thread blocks per increment in each dimension
  Point<DIM> pitches_;
  // Number of concurrent thread blocks
  size_t num_blocks_;
};

template <int32_t DIM>
std::ostream &operator<<(std::ostream &os, const ThreadBlock<DIM> &block)
{
  os << "ThreadBlock(extents: " << block.extents_ << ", pitches: " << block.pitches_ << ")";
  return os;
}

template <int32_t DIM>
std::ostream &operator<<(std::ostream &os, const ThreadBlocks<DIM> &blocks)
{
  os << "ThreadBlocks(" << blocks.block_ << ", extents: " << blocks.extents_
     << ", pitches: " << blocks.pitches_ << ", num concurrent blocks: " << blocks.num_blocks_
     << ", dim order: {";
  for (int32_t dim : blocks.dim_order_) os << dim << ", ";
  os << "})";

  return os;
}

template <typename OP, typename VAL, int32_t DIM>
static __device__ Point<DIM> local_reduce(VAL &result,
                                          AccessorRO<VAL, DIM> in,
                                          VAL identity,
                                          const ThreadBlocks<DIM> &blocks,
                                          const Rect<DIM> &domain,
                                          int32_t collapsed_dim)
{
  const coord_t tid = threadIdx.x;
  const coord_t bid = blockIdx.x;
  Point<DIM> point  = blocks.point(bid, tid, domain.lo);
  if (!domain.contains(point)) return point;

  while (point[collapsed_dim] <= domain.hi[collapsed_dim]) {
    OP::template fold<true>(result, in[point]);
    blocks.next_point(point);
  }

#if __CUDA_ARCH__ >= 700
  // If we're collapsing the innermost dimension, we perform some optimization
  // with shared memory to reduce memory traffic due to atomic updates
  if (collapsed_dim == DIM - 1) {
    __shared__ VAL trampoline[THREADS_PER_BLOCK];
    // Check for the case where all the threads in the same warp have
    // the same x value in which case they're all going to conflict
    // so instead we do a warp-level reduction so just one thread ends
    // up doing the full atomic
    coord_t bucket = 0;
    for (int32_t dim = DIM - 2; dim >= 0; --dim)
      bucket = bucket * (domain.hi[dim + 1] - domain.lo[dim + 1] + 1) + point[dim];

    const uint32_t same_mask = __match_any_sync(0xffffffff, bucket);
    int32_t laneid;
    asm volatile("mov.s32 %0, %laneid;" : "=r"(laneid));
    const uint32_t active_mask = __ballot_sync(0xffffffff, same_mask - (1 << laneid));
    if (active_mask) {
      // Store our data into shared
      trampoline[tid] = result;
      // Make sure all the threads in the warp are done writing
      __syncwarp(active_mask);
      // Have the lowest thread in each mask pull in the values
      int32_t lowest_index = -1;
      for (int32_t i = 0; i < warpSize; i++)
        if (same_mask & (1 << i)) {
          if (lowest_index == -1) {
            if (i != laneid) {
              // We're not the lowest thread in the warp for
              // this value so we're done, set the value back
              // to identity to ensure that we don't try to
              // perform the reduction out to memory
              result = identity;
              break;
            } else  // Make sure we don't do this test again
              lowest_index = i;
            // It was already our value, so just keep going
          } else {
            // Pull in the value from shared memory
            const int32_t index = tid + i - laneid;
            OP::template fold<true>(result, trampoline[index]);
          }
        }
    }
  }
#endif

  return point;
}

template <typename OP, typename VAL, int32_t DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  reduce_with_rw_acc(AccessorRW<VAL, DIM> out,
                     AccessorRO<VAL, DIM> in,
                     VAL identity,
                     ThreadBlocks<DIM> blocks,
                     Rect<DIM> domain,
                     int32_t collapsed_dim)
{
  auto result = identity;
  auto point  = local_reduce<OP, VAL, DIM>(result, in, identity, blocks, domain, collapsed_dim);
  if (result != identity) OP::template fold<false>(out[point], result);
}

template <typename OP, typename VAL, int32_t DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  reduce_with_rd_acc(AccessorRD<OP, false, DIM> out,
                     AccessorRO<VAL, DIM> in,
                     VAL identity,
                     ThreadBlocks<DIM> blocks,
                     Rect<DIM> domain,
                     int32_t collapsed_dim)
{
  auto result = identity;
  auto point  = local_reduce<OP, VAL, DIM>(result, in, identity, blocks, domain, collapsed_dim);
  if (result != identity) out.reduce(point, result);
}

template <UnaryRedCode OP_CODE>
struct UnaryRedImpl {
  template <LegateTypeCode CODE,
            int32_t RHS_DIM,
            std::enable_if_t<(RHS_DIM > 1) && UnaryRedOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(int32_t collapsed_dim,
                  Shape &rhs_shape,
                  RegionField &lhs_rf,
                  RegionField &rhs_rf,
                  bool needs_reduction)
  {
    using OP  = typename UnaryRedOp<OP_CODE, CODE>::OP;
    using VAL = legate_type_of<CODE>;

    auto rhs_rect = rhs_shape.to_rect<RHS_DIM>();
    if (rhs_rect.volume() == 0) return;

    ThreadBlocks<RHS_DIM> blocks;
    blocks.initialize(rhs_rect, collapsed_dim);
    auto rhs = rhs_rf.read_accessor<VAL, RHS_DIM>();

    if (needs_reduction) {
      auto lhs = lhs_rf.reduce_accessor<OP, false, RHS_DIM>();
      blocks.compute_maximum_concurrency(
        reinterpret_cast<const void *>(reduce_with_rd_acc<OP, VAL, RHS_DIM>));
      reduce_with_rd_acc<OP, VAL, RHS_DIM><<<blocks.num_blocks(), blocks.num_threads()>>>(
        lhs, rhs, OP::identity, blocks, rhs_rect, collapsed_dim);
    } else {
      auto lhs = lhs_rf.read_write_accessor<VAL, RHS_DIM>();
      blocks.compute_maximum_concurrency(
        reinterpret_cast<const void *>(reduce_with_rw_acc<OP, VAL, RHS_DIM>));
      reduce_with_rw_acc<OP, VAL, RHS_DIM><<<blocks.num_blocks(), blocks.num_threads()>>>(
        lhs, rhs, OP::identity, blocks, rhs_rect, collapsed_dim);
    }
  }

  template <LegateTypeCode CODE,
            int32_t RHS_DIM,
            std::enable_if_t<RHS_DIM <= 1 || !UnaryRedOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(int32_t collapsed_dim,
                  Shape &rhs_shape,
                  RegionField &lhs_rf,
                  RegionField &rhs_rf,
                  bool needs_reduction)
  {
    assert(false);
  }
};

struct UnaryRedDispatch {
  template <UnaryRedCode OP_CODE>
  void operator()(int32_t collapsed_dim,
                  Shape &rhs_shape,
                  RegionField &lhs,
                  RegionField &rhs,
                  bool needs_reduction)
  {
    return double_dispatch(rhs.dim(),
                           rhs.code(),
                           UnaryRedImpl<OP_CODE>{},
                           collapsed_dim,
                           rhs_shape,
                           lhs,
                           rhs,
                           needs_reduction);
  }
};

}  // namespace gpu

/*static*/ void UnaryRedTask::gpu_variant(const Task *task,
                                          const std::vector<PhysicalRegion> &regions,
                                          Context context,
                                          Runtime *runtime)
{
  Deserializer ctx(task, regions);

  bool needs_reduction;
  int32_t collapsed_dim;
  UnaryRedCode op_code;
  Shape rhs_shape;
  RegionField lhs;
  RegionField rhs;

  deserialize(ctx, needs_reduction);
  deserialize(ctx, collapsed_dim);
  deserialize(ctx, op_code);
  deserialize(ctx, rhs_shape);
  deserialize(ctx, lhs);
  deserialize(ctx, rhs);

  return op_dispatch(
    op_code, gpu::UnaryRedDispatch{}, collapsed_dim, rhs_shape, lhs, rhs, needs_reduction);
}

}  // namespace numpy
}  // namespace legate
