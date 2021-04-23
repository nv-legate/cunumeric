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

#include "shard.h"
#include <cmath>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

using namespace Legion;

namespace legate {
namespace numpy {

/*static*/ NumPyShardingFunctor* NumPyShardingFunctor::sharding_functors[NUMPY_SHARD_LAST];

struct Comparator2D {
 public:
  bool operator()(const std::pair<Point<2>, size_t>& left,
                  const std::pair<Point<2>, size_t>& right) const
  {
    if (left.first[0] < right.first[0])
      return true;
    else if (left.first[0] > right.first[0])
      return false;
    else if (left.first[1] < right.first[1])
      return true;
    else if (left.first[1] > right.first[1])
      return false;
    else
      return (left.second < right.second);
  }
};

// This is a little ugly, but it will allow us to memoize some common
// computations in a way that doesn't require locks
__thread std::map<std::pair<Point<2>, size_t>, std::pair<Point<2>, Point<2>>, Comparator2D>*
  chunks2d = NULL;

static inline Point<2> compute_chunks_2d(const Point<2> bounds, size_t shards, Point<2>& pieces)
{
  const std::pair<Point<2>, size_t> key(bounds, shards);
  // First check to see if we already did this computation
  if (chunks2d != NULL) {
    std::map<std::pair<Point<2>, size_t>, std::pair<Point<2>, Point<2>>, Comparator2D>::
      const_iterator finder = chunks2d->find(key);
    if (finder != chunks2d->end()) {
      pieces = finder->second.first;
      return finder->second.second;
    }
  } else {
    // We're going to need this eventually
    chunks2d =
      new std::map<std::pair<Point<2>, size_t>, std::pair<Point<2>, Point<2>>, Comparator2D>();
  }
  // Didn't find it so we have to compute it
  const bool swap = (bounds[0] > bounds[1]);
  double nx       = swap ? (double)bounds[1] : (double)bounds[0];
  double ny       = swap ? (double)bounds[0] : (double)bounds[1];
  double n        = sqrt(shards * nx / ny);
  // need to constrain n to be an integer with shards % n == 0
  // try rounding n both up and down
  int n1 = floor(n + 1.e-12);
  n1     = MAX(n1, 1);
  while (shards % n1 != 0) --n1;
  int n2 = ceil(n - 1.e-12);
  n2     = MAX(n2, 1);
  while (shards % n2 != 0) ++n2;
  // pick whichever of n1 and n2 gives blocks closest to square,
  // i.e. gives the shortest long side
  double longside1 = MAX(nx / n1, ny / (shards / n1));
  double longside2 = MAX(nx / n2, ny / (shards / n2));
  size_t shardx    = (longside1 <= longside2 ? n1 : n2);
  size_t shardy    = shards / shardx;
  pieces           = Point<2>(swap ? shardy : shardx, swap ? shardx : shardy);
  // Now we can compute the chunk size for each dimension
  Point<2> chunks = (bounds + pieces - Point<2>::ONES()) / pieces;
  chunks[0]       = MAX(chunks[0], coord_t{1});
  chunks[1]       = MAX(chunks[1], coord_t{1});
  // Save the result
  (*chunks2d)[key] = std::pair<Point<2>, Point<2>>(pieces, chunks);
  return chunks;
}

NumPyShardingFunctor::NumPyShardingFunctor(NumPyShardingCode c) : code(c) {}

NumPyShardingFunctor_1D::NumPyShardingFunctor_1D(NumPyShardingCode c) : NumPyShardingFunctor(c) {}

ShardID NumPyShardingFunctor_1D::shard(const DomainPoint& p,
                                       const Domain& launch_space,
                                       const size_t total_shards)
{
  // This one is easy, just chunk it
  const Point<1> point = p;
  const Rect<1> space  = launch_space;
  switch (code) {
    case NUMPY_SHARD_TILE_1D: {
      const size_t size  = (space.hi[0] - space.lo[0]) + 1;
      const size_t chunk = (size + total_shards - 1) / total_shards;
      return (point[0] - space.lo[0]) / chunk;
    }
    default: assert(false);  // not handling any other cases currently
  }
  // Appease the compiler
  return 0;
}

unsigned NumPyShardingFunctor_1D::localize(const DomainPoint& p,
                                           const Domain& launch_space,
                                           const size_t total_shards,
                                           const ShardID local_shard)
{
  // This one is easy, just chunk it
  const Point<1> point = p;
  const Rect<1> space  = launch_space;
  switch (code) {
    case NUMPY_SHARD_TILE_1D: {
      const size_t size   = (space.hi[0] - space.lo[0]) + 1;
      const coord_t chunk = (size + total_shards - 1) / total_shards;
      const coord_t start = local_shard * chunk + space.lo[0];
      assert(point[0] >= start);
      assert(point[0] < (start + chunk));
      return (point[0] - start);
    }
    default: assert(false);  // not handling any other cases currently
  }
  // Appease the compiler
  return 0;
}

NumPyShardingFunctor_2D::NumPyShardingFunctor_2D(NumPyShardingCode c) : NumPyShardingFunctor(c) {}

ShardID NumPyShardingFunctor_2D::shard(const DomainPoint& p,
                                       const Domain& launch_space,
                                       const size_t total_shards)
{
  const Point<2> point  = p;
  const Rect<2> space   = launch_space;
  const Point<2> bounds = (space.hi - space.lo) + Point<2>::ONES();
  Point<2> pieces;
  const Point<2> chunks = compute_chunks_2d(bounds, total_shards, pieces);
  const coord_t x       = (point[0] - space.lo[0]) / chunks[0];
  const coord_t y       = (point[1] - space.lo[1]) / chunks[1];
  switch (code) {
    case NUMPY_SHARD_TILE_2D: {
      // Now linearlize chunk coordinates onto shards
      return (y * pieces[0] + x);
    }
    case NUMPY_SHARD_TILE_2D_YX: {
      // Linearize in the opposite way
      return (x * pieces[1] + y);
    }
    default: assert(false);  // not handling any other cases currently
  }
  // Appease the compiler
  return 0;
}

unsigned NumPyShardingFunctor_2D::localize(const DomainPoint& p,
                                           const Domain& launch_space,
                                           const size_t total_shards,
                                           const ShardID local_shard)
{
  const Point<2> point  = p;
  const Rect<2> space   = launch_space;
  const Point<2> bounds = (space.hi - space.lo) + Point<2>::ONES();
  Point<2> pieces;
  const Point<2> chunks = compute_chunks_2d(bounds, total_shards, pieces);
  switch (code) {
    case NUMPY_SHARD_TILE_2D: {
      const size_t shard_x  = local_shard % pieces[0];
      const size_t shard_y  = local_shard / pieces[0];
      const coord_t start_x = shard_x * chunks[0] + space.lo[0];
      const coord_t start_y = shard_y * chunks[1] + space.lo[1];
      assert(point[0] >= start_x);
      assert(point[0] < (start_x + chunks[0]));
      assert(point[1] >= start_y);
      assert(point[1] < (start_y + chunks[1]));
      return (point[1] - start_y) * chunks[0] + (point[0] - start_x);
    }
    case NUMPY_SHARD_TILE_2D_YX: {
      const size_t shard_y  = local_shard % pieces[1];
      const size_t shard_x  = local_shard / pieces[1];
      const coord_t start_x = shard_x * chunks[0] + space.lo[0];
      const coord_t start_y = shard_y * chunks[1] + space.lo[1];
      assert(point[0] >= start_x);
      assert(point[0] < (start_x + chunks[0]));
      assert(point[1] >= start_y);
      assert(point[1] < (start_y + chunks[1]));
      // Linearize in the opposite way
      return (point[0] - start_x) * chunks[1] + (point[1] - start_y);
    }
    default: assert(false);  // not handling any other cases currently
  }
  // Appease the compiler
  return 0;
}

NumPyShardingFunctor_2D_1D::NumPyShardingFunctor_2D_1D(NumPyShardingCode c)
  : NumPyShardingFunctor(c)
{
}

ShardID NumPyShardingFunctor_2D_1D::shard(const DomainPoint& p,
                                          const Domain& launch_space,
                                          const size_t total_shards)
{
#if 0
      const Point<2> point = p;
      const Rect<2> space = launch_space;
#endif
  switch (code) {
    default: assert(false);  // not handling any other cases currently
  }
  // Appease the compiler
  return 0;
}

unsigned NumPyShardingFunctor_2D_1D::localize(const DomainPoint& p,
                                              const Domain& launch_space,
                                              const size_t total_shards,
                                              const ShardID local_shard)
{
#if 0
      const Point<2> point = p;
      const Rect<2> space = launch_space;
#endif
  switch (code) {
    default: assert(false);  // not handling any other cases currently
  }
  // Appease the compiler
  return 0;
}

NumPyShardingFunctor_3D::NumPyShardingFunctor_3D(NumPyShardingCode c) : NumPyShardingFunctor(c) {}

ShardID NumPyShardingFunctor_3D::shard(const DomainPoint& p,
                                       const Domain& launch_space,
                                       const size_t total_shards)
{
  // TODO: make this properly block things
  // For now linearize and compute
  const Point<3> point = p;
  const Rect<3> space  = launch_space;
  coord_t pitch        = 1;
  coord_t linear_point = 0;
  for (int i = 2; i >= 0; i--) {
    linear_point += point[i] * pitch;
    pitch *= ((space.hi[i] - space.lo[i]) + 1);
  }
  return linear_point % total_shards;
}

unsigned NumPyShardingFunctor_3D::localize(const DomainPoint& p,
                                           const Domain& launch_space,
                                           const size_t total_shards,
                                           const ShardID local_shard)
{
  // TODO: make this properly block things
  // For now linearize and compute
  const Point<3> point = p;
  const Rect<3> space  = launch_space;
  coord_t pitch        = 1;
  coord_t linear_point = 0;
  for (int i = 2; i >= 0; i--) {
    linear_point += point[i] * pitch;
    pitch *= ((space.hi[i] - space.lo[i]) + 1);
  }
  return linear_point / total_shards;
}

NumPyShardingFunctor_3D_2D::NumPyShardingFunctor_3D_2D(NumPyShardingCode c)
  : NumPyShardingFunctor(c)
{
}

ShardID NumPyShardingFunctor_3D_2D::shard(const DomainPoint& p,
                                          const Domain& launch_space,
                                          const size_t total_shards)
{
  const Point<3> point = p;
  const Rect<3> space  = launch_space;
  switch (code) {
    case NUMPY_SHARD_TILE_3D_2D_XY: {
      const Point<2> bounds((space.hi[0] - space.lo[0]) + 1, (space.hi[1] - space.lo[1]) + 1);
      Point<2> pieces;
      const Point<2> chunks = compute_chunks_2d(bounds, total_shards, pieces);
      const coord_t x       = (point[0] - space.lo[0]) / chunks[0];
      const coord_t y       = (point[1] - space.lo[1]) / chunks[1];
      // Now linearlize chunk coordinates onto shards
      return (y * pieces[0] + x);
    }
    case NUMPY_SHARD_TILE_3D_2D_XZ: {
      const Point<2> bounds((space.hi[0] - space.lo[0]) + 1, (space.hi[2] - space.lo[2]) + 1);
      Point<2> pieces;
      const Point<2> chunks = compute_chunks_2d(bounds, total_shards, pieces);
      const coord_t x       = (point[0] - space.lo[0]) / chunks[0];
      const coord_t y       = (point[2] - space.lo[2]) / chunks[1];
      // Now linearlize chunk coordinates onto shards
      return (y * pieces[0] + x);
    }
    case NUMPY_SHARD_TILE_3D_2D_YZ: {
      const Point<2> bounds((space.hi[1] - space.lo[1]) + 1, (space.hi[2] - space.lo[2]) + 1);
      Point<2> pieces;
      const Point<2> chunks = compute_chunks_2d(bounds, total_shards, pieces);
      const coord_t x       = (point[1] - space.lo[1]) / chunks[0];
      const coord_t y       = (point[2] - space.lo[2]) / chunks[1];
      // Now linearlize chunk coordinates onto shards
      return (y * pieces[0] + x);
    }
    default: assert(false);
  }
  // Appease the compiler
  return 0;
}

unsigned NumPyShardingFunctor_3D_2D::localize(const DomainPoint& p,
                                              const Domain& launch_space,
                                              const size_t total_shards,
                                              const ShardID local_shard)
{
  const Point<3> point = p;
  const Rect<3> space  = launch_space;
  switch (code) {
    case NUMPY_SHARD_TILE_3D_2D_XY: {
      const Point<2> bounds((space.hi[0] - space.lo[0]) + 1, (space.hi[1] - space.lo[1]) + 1);
      Point<2> pieces;
      const Point<2> chunks = compute_chunks_2d(bounds, total_shards, pieces);
      const size_t shard_x  = local_shard % pieces[0];
      const size_t shard_y  = local_shard / pieces[0];
      const coord_t start_x = shard_x * chunks[0] + space.lo[0];
      const coord_t start_y = shard_y * chunks[1] + space.lo[1];
      assert(point[0] >= start_x);
      assert(point[0] < (start_x + chunks[0]));
      assert(point[1] >= start_y);
      assert(point[1] < (start_y + chunks[1]));
      return (point[1] - start_y) * chunks[0] + (point[0] - start_x);
    }
    case NUMPY_SHARD_TILE_3D_2D_XZ: {
      const Point<2> bounds((space.hi[0] - space.lo[0]) + 1, (space.hi[2] - space.lo[2]) + 1);
      Point<2> pieces;
      const Point<2> chunks = compute_chunks_2d(bounds, total_shards, pieces);
      const size_t shard_x  = local_shard % pieces[0];
      const size_t shard_y  = local_shard / pieces[0];
      const coord_t start_x = shard_x * chunks[0] + space.lo[0];
      const coord_t start_y = shard_y * chunks[1] + space.lo[2];
      assert(point[0] >= start_x);
      assert(point[0] < (start_x + chunks[0]));
      assert(point[2] >= start_y);
      assert(point[2] < (start_y + chunks[1]));
      return (point[2] - start_y) * chunks[0] + (point[0] - start_x);
    }
    case NUMPY_SHARD_TILE_3D_2D_YZ: {
      const Point<2> bounds((space.hi[1] - space.lo[1]) + 1, (space.hi[2] - space.lo[2]) + 1);
      Point<2> pieces;
      const Point<2> chunks = compute_chunks_2d(bounds, total_shards, pieces);
      const size_t shard_x  = local_shard % pieces[0];
      const size_t shard_y  = local_shard / pieces[0];
      const coord_t start_x = shard_x * chunks[0] + space.lo[1];
      const coord_t start_y = shard_y * chunks[1] + space.lo[2];
      assert(point[1] >= start_x);
      assert(point[1] < (start_x + chunks[0]));
      assert(point[2] >= start_y);
      assert(point[2] < (start_y + chunks[1]));
      return (point[2] - start_y) * chunks[0] + (point[1] - start_x);
    }
    default: assert(false);
  }
  // Appease the compiler
  return 0;
}

template <int DIM, int RADIX>
NumPyShardingFunctor_Radix2D<DIM, RADIX>::NumPyShardingFunctor_Radix2D(NumPyShardingCode c)
  : NumPyShardingFunctor(c)
{
  assert(RADIX > 0);
}

template <int DIM, int RADIX>
ShardID NumPyShardingFunctor_Radix2D<DIM, RADIX>::shard(const DomainPoint& p,
                                                        const Domain& launch_space,
                                                        const size_t total_shards)
{
  const Point<2> point = p;
  const Rect<2> space  = launch_space;
  // See if this is an inner product case or not
  if (space.lo[(DIM + 1) % 2] == space.hi[(DIM + 1) % 2]) {
    // Inner product case
    assert(point[(DIM + 1) % 2] == 0);
    const coord_t num_pieces = (space.hi[DIM] - space.lo[DIM]) + 1;
    const coord_t chunk      = (num_pieces + total_shards - 1) / total_shards;
    return point[DIM] / chunk;
  } else {
    // Normal case
    const coord_t num_pieces = (space.hi[(DIM + 1) % 2] - space.lo[(DIM + 1) % 2]) + 1;
    const coord_t orig_n     = (space.hi[DIM] - space.lo[DIM]) + 1;
    assert((num_pieces % orig_n) == 0);
    const coord_t orig_m = num_pieces / orig_n;
    // These are our coordinates on the old sharding so figure out
    // how they got mapped down to shards before
    const Point<2> bounds(orig_m, orig_n);
    Point<2> pieces;
    const Point<2> chunks = compute_chunks_2d(bounds, total_shards, pieces);
    // Figure out how to map our coordinates onto the old shard
    const coord_t m = point[(DIM + 1) % 2] / orig_n;
    assert(m < orig_m);
    const coord_t n = point[DIM % 2] * RADIX + ((point[(DIM + 1) % 2] % orig_n) % RADIX);
    assert(n < orig_n);
    const coord_t x = m / chunks[0];
    const coord_t y = n / chunks[1];
    return (y * pieces[0] + x);
  }
}

template <int DIM, int RADIX>
unsigned NumPyShardingFunctor_Radix2D<DIM, RADIX>::localize(const DomainPoint& p,
                                                            const Domain& launch_space,
                                                            const size_t total_shards,
                                                            const ShardID local_shard)
{
  const Point<2> point = p;
  const Rect<2> space  = launch_space;
  // See if this is an inner product case or not
  if (space.lo[(DIM + 1) % 2] == space.hi[(DIM + 1) % 2]) {
    // Inner product case
    assert(point[(DIM + 1) % 2] == 0);
    const coord_t num_pieces = (space.hi[DIM] - space.lo[DIM]) + 1;
    const coord_t chunk      = (num_pieces + total_shards - 1) / total_shards;
    assert((local_shard * chunk) <= point[DIM]);
    assert(point[DIM] < ((local_shard + 1) * chunk));
    // Compute the offset within our local shard
    return (point[DIM] - local_shard * chunk);
  } else {
    // Normal case
    const coord_t num_pieces = (space.hi[(DIM + 1) % 2] - space.lo[(DIM + 1) % 2]) + 1;
    const coord_t orig_n     = (space.hi[DIM] - space.lo[DIM]) + 1;
    // assert((num_pieces % orig_n) == 0);
    const coord_t orig_m = num_pieces / orig_n;
    // These are our coordinates on the old sharding so figure out
    // how they got mapped down to shards before
    const Point<2> bounds(orig_m, orig_n);
    Point<2> pieces;
    const Point<2> chunks = compute_chunks_2d(bounds, total_shards, pieces);
    const coord_t m       = point[(DIM + 1) % 2] / orig_n;
    // assert(m < orig_m);
    const coord_t n = point[DIM % 2] * RADIX + ((point[(DIM + 1) % 2] % orig_n) % RADIX);
    // assert(n < orig_n);
    const coord_t x = m % chunks[0];
    const coord_t y = n % chunks[1];
    return y * chunks[0] + x;
  }
}

template <int DIM, int RADIX>
NumPyShardingFunctor_Radix3D<DIM, RADIX>::NumPyShardingFunctor_Radix3D(NumPyShardingCode c)
  : NumPyShardingFunctor(c)
{
  assert(RADIX > 0);
}

template <int DIM, int RADIX>
ShardID NumPyShardingFunctor_Radix3D<DIM, RADIX>::shard(const DomainPoint& p,
                                                        const Domain& launch_space,
                                                        const size_t total_shards)
{
  const Point<3> point = p;
  const Rect<3> space  = launch_space;
  // See if this is an inner product case or not
  if ((space.lo[(DIM + 1) % 3] == space.hi[(DIM + 1) % 3]) &&
      (space.lo[(DIM + 2) % 3] == space.hi[(DIM + 2) % 3])) {
    // Inner-product case
    const coord_t num_pieces = (space.hi[DIM] - space.lo[DIM]) + 1;
    const coord_t chunk      = (num_pieces + total_shards - 1) / total_shards;
    return point[DIM] / chunk;
  } else {
    // Normal case
    const coord_t collapsed_size = (space.hi[DIM] - space.lo[DIM]) + 1;
    const coord_t one_size       = (space.hi[(DIM + 1) % 3] - space.lo[(DIM + 1) % 3]) + 1;
    const coord_t two_size       = (space.hi[(DIM + 2) % 3] - space.lo[(DIM + 2) % 3]) + 1;
    // One of the two other sizes should be the same as the collapsing one
    assert((one_size == collapsed_size) || (two_size == collapsed_size));
    // Figure out which dimension is partitioned the same as the collapsing dim
    const Point<2> bounds((DIM == 1) ? two_size : one_size, (DIM == 1) ? one_size : two_size);
    Point<2> pieces;
    const Point<2> chunks = compute_chunks_2d(bounds, total_shards, pieces);
    if (DIM == 0) {
      if (one_size == collapsed_size) {
        const coord_t x = point[0] * RADIX / chunks[0];
        const coord_t y = point[1] / chunks[0];
        const coord_t z = point[2] / chunks[1];
        return z * pieces[0] + x + (y % RADIX);
      } else {
        const coord_t x = point[0] * RADIX / chunks[1];
        const coord_t y = point[1] / chunks[0];
        const coord_t z = point[2] / chunks[1];
        return (x + (z % RADIX)) * pieces[0] + y;
      }
    } else if (DIM == 1) {
      if (one_size == collapsed_size) {
        const coord_t x = point[0] / chunks[0];
        const coord_t y = point[1] * RADIX / chunks[0];
        const coord_t z = point[2] / chunks[1];
        return z * pieces[0] + y + (x % RADIX);
      } else {
        const coord_t x = point[0] / chunks[0];
        const coord_t y = point[1] * RADIX / chunks[1];
        const coord_t z = point[2] / chunks[1];
        return (y + (z % RADIX)) * pieces[0] + x;
      }
    } else if (DIM == 2) {
      if (one_size == collapsed_size) {
        const coord_t x = point[0] / chunks[0];
        const coord_t y = point[1] / chunks[1];
        const coord_t z = point[2] * RADIX / chunks[0];
        return y * pieces[0] + z + (x % RADIX);
      } else {
        const coord_t x = point[0] / chunks[0];
        const coord_t y = point[1] / chunks[1];
        const coord_t z = point[2] * RADIX / chunks[1];
        return (z + (y % RADIX)) * pieces[0] + x;
      }
    }
  }
  // Should never get here
  assert(false);
  return 0;
}

template <int DIM, int RADIX>
unsigned NumPyShardingFunctor_Radix3D<DIM, RADIX>::localize(const DomainPoint& p,
                                                            const Domain& launch_space,
                                                            const size_t total_shards,
                                                            const ShardID local_shard)
{
  const Point<3> point = p;
  const Rect<3> space  = launch_space;
  // See if this is an inner product case or not
  if ((space.lo[(DIM + 1) % 3] == space.hi[(DIM + 1) % 3]) &&
      (space.lo[(DIM + 2) % 3] == space.hi[(DIM + 2) % 3])) {
    // Inner product case
    const coord_t num_pieces = (space.hi[DIM] - space.lo[DIM]) + 1;
    const coord_t chunk      = (num_pieces + total_shards - 1) / total_shards;
    assert((local_shard * chunk) <= point[DIM]);
    assert(point[DIM] < ((local_shard + 1) * chunk));
    // Compute the offset within our local shard
    return (point[DIM] - local_shard * chunk);
  } else {
    // Normal case
    const coord_t collapsed_size = (space.hi[DIM] - space.lo[DIM]) + 1;
    const coord_t one_size       = (space.hi[(DIM + 1) % 3] - space.lo[(DIM + 1) % 3]) + 1;
    const coord_t two_size       = (space.hi[(DIM + 2) % 3] - space.lo[(DIM + 2) % 3]) + 1;
    // One of the two other sizes should be the same as the collapsing one
    // MZ: commenting out this assert for test correctness, but this needs to be looked at further
    // assert((one_size == collapsed_size) || (two_size == collapsed_size));
    // Figure out which dimension is partitioned the same as the collapsing dim
    const Point<2> bounds((DIM == 1) ? two_size : one_size, (DIM == 1) ? one_size : two_size);
    Point<2> pieces;
    const Point<2> chunks = compute_chunks_2d(bounds, total_shards, pieces);
    if (DIM == 0) {
      if (one_size == collapsed_size) {
        const coord_t x = (point[0] * RADIX) % chunks[0];
        const coord_t y = point[1] % chunks[0];
        const coord_t z = point[2] % chunks[1];
        return z * chunks[0] + x + (y % RADIX);
      } else {
        const coord_t x = (point[0] * RADIX) % chunks[1];
        const coord_t y = point[1] % chunks[0];
        const coord_t z = point[2] % chunks[1];
        return (x + (z % RADIX)) * chunks[0] + y;
      }
    } else if (DIM == 1) {
      if (one_size == collapsed_size) {
        const coord_t x = point[0] % chunks[0];
        const coord_t y = (point[1] * RADIX) % chunks[0];
        const coord_t z = point[2] / chunks[1];
        return z * chunks[0] + y + (x % RADIX);
      } else {
        const coord_t x = point[0] % chunks[0];
        const coord_t y = (point[1] * RADIX) % chunks[1];
        const coord_t z = point[2] % chunks[1];
        return (y + (z % RADIX)) * chunks[0] + x;
      }
    } else if (DIM == 2) {
      if (one_size == collapsed_size) {
        const coord_t x = point[0] % chunks[0];
        const coord_t y = point[1] % chunks[1];
        const coord_t z = (point[2] * RADIX) % chunks[0];
        return y * chunks[0] + z + (x % RADIX);
      } else {
        const coord_t x = point[0] % chunks[0];
        const coord_t y = point[1] % chunks[1];
        const coord_t z = (point[2] * RADIX) % chunks[1];
        return (z + (y % RADIX)) * chunks[0] + x;
      }
    }
  }
  // Should never get here
  assert(false);
  return 0;
}

template <int M, typename BASE>
NumPyTransformShardingFunctor<M, BASE>::NumPyTransformShardingFunctor(NumPyShardingCode code,
                                                                      const long* data,
                                                                      unsigned n)
  : BASE(code), N(n), transform((long*)malloc(M * (N + 1) * sizeof(long)))
{
  memcpy(transform, data, M * (N + 1) * sizeof(long));
}

template <int M, typename BASE>
NumPyTransformShardingFunctor<M, BASE>::~NumPyTransformShardingFunctor(void)
{
  free(transform);
}

template <int M, typename BASE>
ShardID NumPyTransformShardingFunctor<M, BASE>::shard(const DomainPoint& p,
                                                      const Domain& launch_space,
                                                      const size_t total_shards)
{
  assert(p.get_dim() == N);
  assert(launch_space.get_dim() == M);
  DomainPoint point;
  point.dim = M;
  for (unsigned i = 0; i < M; i++) {
    point.point_data[i] = transform[i * (N + 1) + N];  // offset
    for (unsigned j = 0; j < N; j++)
      point.point_data[i] += transform[i * (N + 1) + j] * p.point_data[j];
  }
  return BASE::shard(point, launch_space, total_shards);
}

template <int M, typename BASE>
unsigned NumPyTransformShardingFunctor<M, BASE>::localize(const DomainPoint& p,
                                                          const Domain& launch_space,
                                                          size_t total_shards,
                                                          const ShardID local_shard)
{
  assert(p.get_dim() == N);
  assert(launch_space.get_dim() == M);
  DomainPoint point;
  point.dim = M;
  for (unsigned i = 0; i < M; i++) {
    point.point_data[i] = transform[i * (N + 1) + N];  // offset
    for (unsigned j = 0; j < N; j++)
      point.point_data[i] += transform[i * (N + 1) + j] * p.point_data[j];
  }
  return BASE::localize(point, launch_space, total_shards, local_shard);
}

// Some template help for unrolling
template <int X, int P>
struct Pow {
  enum { result = X * Pow<X, P - 1>::result };
};

template <int X>
struct Pow<X, 0> {
  enum { result = 1 };
};

template <int X>
struct Pow<X, 1> {
  enum { result = X };
};

template <typename T>
static void register_functor(Runtime* runtime, ShardingID offset, NumPyShardingCode code)
{
  T* functor = new T(code);
  runtime->register_sharding_functor(
    (ShardingID)(offset + code), functor, true /*silence warnings*/);
  // Save this is in the mapper sharding functors array
  assert(code < NUMPY_SHARD_LAST);
  NumPyShardingFunctor::sharding_functors[code] = functor;
}

/*static*/ void NumPyShardingFunctor::register_sharding_functors(Runtime* runtime,
                                                                 ShardingID offset)
{
  register_functor<NumPyShardingFunctor_1D>(runtime, offset, NUMPY_SHARD_TILE_1D);
  register_functor<NumPyShardingFunctor_2D>(runtime, offset, NUMPY_SHARD_TILE_2D);
  register_functor<NumPyShardingFunctor_3D>(runtime, offset, NUMPY_SHARD_TILE_3D);
  register_functor<NumPyShardingFunctor_2D>(runtime, offset, NUMPY_SHARD_TILE_2D_YX);

  register_functor<NumPyShardingFunctor_3D_2D>(runtime, offset, NUMPY_SHARD_TILE_3D_2D_XY);
  register_functor<NumPyShardingFunctor_3D_2D>(runtime, offset, NUMPY_SHARD_TILE_3D_2D_XZ);
  register_functor<NumPyShardingFunctor_3D_2D>(runtime, offset, NUMPY_SHARD_TILE_3D_2D_YZ);

  register_functor<NumPyShardingFunctor_Radix2D<0, Pow<NUMPY_RADIX, 1>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_2D_X_1);
  register_functor<NumPyShardingFunctor_Radix2D<0, Pow<NUMPY_RADIX, 2>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_2D_X_2);
  register_functor<NumPyShardingFunctor_Radix2D<0, Pow<NUMPY_RADIX, 3>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_2D_X_3);
  register_functor<NumPyShardingFunctor_Radix2D<0, Pow<NUMPY_RADIX, 4>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_2D_X_4);
  register_functor<NumPyShardingFunctor_Radix2D<0, Pow<NUMPY_RADIX, 5>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_2D_X_5);
  register_functor<NumPyShardingFunctor_Radix2D<0, Pow<NUMPY_RADIX, 6>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_2D_X_6);
  register_functor<NumPyShardingFunctor_Radix2D<0, Pow<NUMPY_RADIX, 7>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_2D_X_7);
  register_functor<NumPyShardingFunctor_Radix2D<0, Pow<NUMPY_RADIX, 8>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_2D_X_8);

  register_functor<NumPyShardingFunctor_Radix2D<1, Pow<NUMPY_RADIX, 1>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_2D_Y_1);
  register_functor<NumPyShardingFunctor_Radix2D<1, Pow<NUMPY_RADIX, 2>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_2D_Y_2);
  register_functor<NumPyShardingFunctor_Radix2D<1, Pow<NUMPY_RADIX, 3>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_2D_Y_3);
  register_functor<NumPyShardingFunctor_Radix2D<1, Pow<NUMPY_RADIX, 4>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_2D_Y_4);
  register_functor<NumPyShardingFunctor_Radix2D<1, Pow<NUMPY_RADIX, 5>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_2D_Y_5);
  register_functor<NumPyShardingFunctor_Radix2D<1, Pow<NUMPY_RADIX, 6>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_2D_Y_6);
  register_functor<NumPyShardingFunctor_Radix2D<1, Pow<NUMPY_RADIX, 7>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_2D_Y_7);
  register_functor<NumPyShardingFunctor_Radix2D<1, Pow<NUMPY_RADIX, 8>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_2D_Y_8);

  register_functor<NumPyShardingFunctor_Radix3D<0, Pow<NUMPY_RADIX, 1>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_X_1);
  register_functor<NumPyShardingFunctor_Radix3D<0, Pow<NUMPY_RADIX, 2>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_X_2);
  register_functor<NumPyShardingFunctor_Radix3D<0, Pow<NUMPY_RADIX, 3>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_X_3);
  register_functor<NumPyShardingFunctor_Radix3D<0, Pow<NUMPY_RADIX, 4>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_X_4);
  register_functor<NumPyShardingFunctor_Radix3D<0, Pow<NUMPY_RADIX, 5>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_X_5);
  register_functor<NumPyShardingFunctor_Radix3D<0, Pow<NUMPY_RADIX, 6>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_X_6);
  register_functor<NumPyShardingFunctor_Radix3D<0, Pow<NUMPY_RADIX, 7>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_X_7);
  register_functor<NumPyShardingFunctor_Radix3D<0, Pow<NUMPY_RADIX, 8>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_X_8);

  register_functor<NumPyShardingFunctor_Radix3D<1, Pow<NUMPY_RADIX, 1>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_Y_1);
  register_functor<NumPyShardingFunctor_Radix3D<1, Pow<NUMPY_RADIX, 2>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_Y_2);
  register_functor<NumPyShardingFunctor_Radix3D<1, Pow<NUMPY_RADIX, 3>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_Y_3);
  register_functor<NumPyShardingFunctor_Radix3D<1, Pow<NUMPY_RADIX, 4>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_Y_4);
  register_functor<NumPyShardingFunctor_Radix3D<1, Pow<NUMPY_RADIX, 5>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_Y_5);
  register_functor<NumPyShardingFunctor_Radix3D<1, Pow<NUMPY_RADIX, 6>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_Y_6);
  register_functor<NumPyShardingFunctor_Radix3D<1, Pow<NUMPY_RADIX, 7>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_Y_7);
  register_functor<NumPyShardingFunctor_Radix3D<1, Pow<NUMPY_RADIX, 8>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_Y_8);

  register_functor<NumPyShardingFunctor_Radix3D<2, Pow<NUMPY_RADIX, 1>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_Z_1);
  register_functor<NumPyShardingFunctor_Radix3D<2, Pow<NUMPY_RADIX, 2>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_Z_2);
  register_functor<NumPyShardingFunctor_Radix3D<2, Pow<NUMPY_RADIX, 3>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_Z_3);
  register_functor<NumPyShardingFunctor_Radix3D<2, Pow<NUMPY_RADIX, 4>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_Z_4);
  register_functor<NumPyShardingFunctor_Radix3D<2, Pow<NUMPY_RADIX, 5>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_Z_5);
  register_functor<NumPyShardingFunctor_Radix3D<2, Pow<NUMPY_RADIX, 6>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_Z_6);
  register_functor<NumPyShardingFunctor_Radix3D<2, Pow<NUMPY_RADIX, 7>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_Z_7);
  register_functor<NumPyShardingFunctor_Radix3D<2, Pow<NUMPY_RADIX, 8>::result>>(
    runtime, offset, NUMPY_SHARD_RADIX_3D_Z_8);
}

}  // namespace numpy
}  // namespace legate

extern "C" {

void legate_numpy_create_transform_sharding_functor(
  unsigned first, unsigned offset, unsigned M, unsigned N, const long* transform)
{
  legate::numpy::NumPyShardingFunctor* functor = NULL;
  switch (M) {
    case 1: {
      functor =
        new legate::numpy::NumPyTransformShardingFunctor<1, legate::numpy::NumPyShardingFunctor_1D>(
          NUMPY_SHARD_TILE_1D, transform, N);
      break;
    }
    case 2: {
      functor =
        new legate::numpy::NumPyTransformShardingFunctor<2, legate::numpy::NumPyShardingFunctor_2D>(
          NUMPY_SHARD_TILE_2D, transform, N);
      break;
    }
    case 3: {
      functor =
        new legate::numpy::NumPyTransformShardingFunctor<3, legate::numpy::NumPyShardingFunctor_3D>(
          NUMPY_SHARD_TILE_3D, transform, N);
      break;
    }
    default: assert(false);  // should never get here
  }
  Runtime* runtime = Runtime::get_runtime();
  runtime->register_sharding_functor(first + offset, functor, true /*silence warnings*/);
  // Also record this as a sharding functor for the mappers
  legate::numpy::NumPyShardingFunctor::sharding_functors[offset] = functor;
}
}
