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

#include "unary_red.h"
#include "unary_red_util.h"
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
  __host__ __device__ Point<DIM> point(coord_t tid)
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
  __host__ __device__ Point<DIM> point(coord_t bid, coord_t tid, const Point<DIM> &origin)
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

template <typename T>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  dense_init_kernel(size_t volume, T *out, T init)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  out[idx] = init;
}

template <typename WriteAcc, typename T, typename Pitches, typename Point>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  generic_init_kernel(size_t volume, WriteAcc out, T init, Pitches pitches, Point lo)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto point = pitches.unflatten(idx, lo);
  out[point] = init;
}

template <typename Op, typename VAL, int32_t DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  red_kernel(Op op,
             AccessorWO<VAL, DIM> out,
             AccessorRO<VAL, DIM> in,
             VAL identity,
             ThreadBlocks<DIM> blocks,
             Rect<DIM> domain,
             int32_t collapsed_dim)
{
  coord_t tid      = threadIdx.x;
  coord_t bid      = blockIdx.x;
  Point<DIM> point = blocks.point(bid, tid, domain.lo);
  if (!domain.contains(point)) return;

  auto result = identity;
  while (point[collapsed_dim] <= domain.hi[collapsed_dim]) {
    Op::template fold<true>(result, in[point]);
    blocks.next_point(point);
  }

  if (result != identity) Op::template fold<false>(out[point], result);
}

template <UnaryRedCode OP_CODE>
struct UnaryRedImpl {
  template <LegateTypeCode CODE,
            int32_t RHS_DIM,
            std::enable_if_t<(RHS_DIM > 1) && UnaryRedOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(int32_t collapsed_dim,
                  Shape &lhs_shape,
                  Shape &rhs_shape,
                  RegionField &lhs_init_rf,
                  RegionField &lhs_red_rf,
                  RegionField &rhs_rf)
  {
    constexpr int32_t LHS_DIM = RHS_DIM - 1;
    using OP                  = UnaryRedOp<OP_CODE, CODE>;
    using VAL                 = legate_type_of<CODE>;

    Pitches<LHS_DIM - 1> lhs_pitches;
    auto lhs_rect     = lhs_shape.to_rect<LHS_DIM>();
    size_t lhs_volume = lhs_pitches.flatten(lhs_rect);

    if (lhs_volume == 0) return;

    auto lhs_init = lhs_init_rf.write_accessor<VAL, LHS_DIM>();
#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = lhs_init.accessor.is_dense_row_major(lhs_rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    const size_t lhs_blocks = (lhs_volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dense) {
      auto lhs = lhs_init.ptr(lhs_rect);
      dense_init_kernel<<<lhs_blocks, THREADS_PER_BLOCK>>>(lhs_volume, lhs, OP::identity);
    } else {
      generic_init_kernel<<<lhs_blocks, THREADS_PER_BLOCK>>>(
        lhs_volume, lhs_init, OP::identity, lhs_pitches, lhs_rect.lo);
    }

    ThreadBlocks<RHS_DIM> blocks;
    auto rhs_rect = rhs_shape.to_rect<RHS_DIM>();
    blocks.initialize(rhs_rect, collapsed_dim);
    blocks.compute_maximum_concurrency(
      reinterpret_cast<const void *>(red_kernel<OP, VAL, RHS_DIM>));
    auto lhs_red = lhs_red_rf.write_accessor<VAL, RHS_DIM>();
    auto rhs     = rhs_rf.read_accessor<VAL, RHS_DIM>();

    red_kernel<<<blocks.num_blocks(), blocks.num_threads()>>>(
      OP{}, lhs_red, rhs, OP::identity, blocks, rhs_rect, collapsed_dim);
  }

  template <LegateTypeCode CODE,
            int32_t RHS_DIM,
            std::enable_if_t<RHS_DIM <= 1 || !UnaryRedOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(int32_t collapsed_dim,
                  Shape &lhs_shape,
                  Shape &rhs_shape,
                  RegionField &lhs_init,
                  RegionField &lhs_red,
                  RegionField &rhs)
  {
    assert(false);
  }
};

struct UnaryRedDispatch {
  template <UnaryRedCode OP_CODE>
  void operator()(int32_t collapsed_dim,
                  Shape &lhs_shape,
                  Shape &rhs_shape,
                  RegionField &lhs_init,
                  RegionField &lhs_red,
                  RegionField &rhs)
  {
    return double_dispatch(rhs.dim(),
                           rhs.code(),
                           UnaryRedImpl<OP_CODE>{},
                           collapsed_dim,
                           lhs_shape,
                           rhs_shape,
                           lhs_init,
                           lhs_red,
                           rhs);
  }
};

}  // namespace gpu

/*static*/ void UnaryRedTask::gpu_variant(const Task *task,
                                          const std::vector<PhysicalRegion> &regions,
                                          Context context,
                                          Runtime *runtime)
{
  Deserializer ctx(task, regions);

  int32_t collapsed_dim;
  UnaryRedCode op_code;
  Shape lhs_shape;
  Shape rhs_shape;
  // out_init and out_red are aliases of the same region field but with different transformations
  RegionField lhs_init;
  RegionField lhs_red;
  RegionField rhs;

  deserialize(ctx, collapsed_dim);
  deserialize(ctx, op_code);
  deserialize(ctx, lhs_shape);
  deserialize(ctx, rhs_shape);
  deserialize(ctx, lhs_init);
  deserialize(ctx, lhs_red);
  deserialize(ctx, rhs);

  return op_dispatch(
    op_code, gpu::UnaryRedDispatch{}, collapsed_dim, lhs_shape, rhs_shape, lhs_init, lhs_red, rhs);
}

}  // namespace numpy
}  // namespace legate
