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
template <int DIM>
struct ThreadBlock {
  void initialize(const Rect<DIM> &domain, int collapsed_dim)
  {
    auto remaining = static_cast<coord_t>(THREADS_PER_BLOCK);

    Point<DIM> domain_extents;
    for (int idx = 0; idx < DIM; ++idx) domain_extents[idx] = domain.hi[idx] - domain.lo[idx] + 1;

    if (collapsed_dim == DIM - 1) {
      auto extent             = std::min<coord_t>(WARP_SIZE, domain_extents[collapsed_dim]);
      extents_[collapsed_dim] = extent;
      remaining               = std::max<coord_t>(remaining / extent, 1);
    }

    for (int idx = DIM - 1; idx >= 0; --idx) {
      if (idx == collapsed_dim) continue;
      auto extent   = std::min(remaining, domain_extents[idx]);
      extents_[idx] = extent;
      remaining     = std::max<coord_t>(remaining / extent, 1);
    }

    if (collapsed_dim != DIM - 1)
      extents_[collapsed_dim] = std::min(remaining, domain_extents[collapsed_dim]);

    num_threads_ = 1;
    for (int idx = DIM - 1; idx >= 0; --idx) {
      pitches_[idx] = num_threads_;
      num_threads_ *= extents_[idx];
    }
  }

  __host__ __device__ Point<DIM> point(coord_t tid)
  {
    Point<DIM> p;
    for (int dim = 0; dim < DIM; ++dim) {
      p[dim] = tid / pitches_[dim];
      tid    = tid % pitches_[dim];
    }
    return p;
  }

  Point<DIM> extents_;
  Point<DIM> pitches_;
  size_t num_threads_;
};

template <int DIM>
struct ThreadBlocks {
  void initialize(const Rect<DIM> &domain, int collapsed_dim)
  {
    block_.initialize(domain, collapsed_dim);

    for (int idx = 0; idx < DIM; ++idx) {
      auto domain_extent = domain.hi[idx] - domain.lo[idx] + 1;
      extents_[idx]      = div_and_ceil(domain_extent, block_.extents_[idx]);
    }

    num_blocks_ = 1;
    for (int idx = DIM - 1; idx >= 0; --idx) {
      pitches_[idx] = num_blocks_;
      num_blocks_ *= extents_[idx];
    }
  }

  __host__ __device__ Point<DIM> point(coord_t bid, coord_t tid, const Point<DIM> &origin)
  {
    Point<DIM> p = origin;
    for (int dim = 0; dim < DIM; ++dim) {
      p[dim] += (bid / pitches_[dim]) * block_.extents_[dim];
      bid = bid % pitches_[dim];
    }
    p += block_.point(tid);
    return p;
  }

  constexpr size_t num_blocks() const { return num_blocks_; }
  constexpr size_t num_threads() const { return block_.num_threads_; }

  ThreadBlock<DIM> block_;
  Point<DIM> extents_;
  Point<DIM> pitches_;
  size_t num_blocks_;
};

template <int DIM>
std::ostream &operator<<(std::ostream &os, const ThreadBlock<DIM> &block)
{
  os << "ThreadBlock(extents: " << block.extents_ << ", pitches: " << block.pitches_ << ")";
  return os;
}

template <int DIM>
std::ostream &operator<<(std::ostream &os, const ThreadBlocks<DIM> &block)
{
  os << "ThreadBlocks(" << block.block_ << ", extents: " << block.extents_
     << ", pitches: " << block.pitches_ << ")";
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

template <typename Op,
          typename WriteAcc,
          typename ReadAcc,
          typename T,
          typename ThreadBlocks,
          typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  red_kernel(Op op, WriteAcc out, ReadAcc in, T identity, ThreadBlocks blocks, Rect domain)
{
  coord_t tid = threadIdx.x;
  coord_t bid = blockIdx.x;
  auto point  = blocks.point(bid, tid, domain.lo);

  if (!domain.contains(point)) return;

  auto result = identity;

  Op::template fold<true>(result, in[point]);

  Op::template fold<false>(out[point], result);
}

template <UnaryRedCode OP_CODE>
struct UnaryRedImpl {
  template <LegateTypeCode CODE,
            int RHS_DIM,
            std::enable_if_t<(RHS_DIM > 1) && UnaryRedOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(int collapsed_dim,
                  Shape &lhs_shape,
                  Shape &rhs_shape,
                  RegionField &lhs_init_rf,
                  RegionField &lhs_red_rf,
                  RegionField &rhs_rf)
  {
    constexpr int LHS_DIM = RHS_DIM - 1;
    using OP              = UnaryRedOp<OP_CODE, CODE>;
    using VAL             = legate_type_of<CODE>;

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
    auto lhs_red = lhs_red_rf.write_accessor<VAL, RHS_DIM>();
    auto rhs     = rhs_rf.read_accessor<VAL, RHS_DIM>();

    red_kernel<<<blocks.num_blocks(), blocks.num_threads()>>>(
      OP{}, lhs_red, rhs, OP::identity, blocks, rhs_rect);
  }

  template <LegateTypeCode CODE,
            int RHS_DIM,
            std::enable_if_t<RHS_DIM <= 1 || !UnaryRedOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(int collapsed_dim,
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
  void operator()(int collapsed_dim,
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
