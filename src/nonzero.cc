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

#include "nonzero.h"
#include "proj.h"
#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>
#ifdef LEGATE_USE_OPENMP
#include <alloca.h>
#include <omp.h>
#endif

using namespace Legion;

namespace legate {
namespace numpy {

template <typename T>
/*static*/ void CountNonzeroTask<T>::cpu_variant(const Task* task,
                                                 const std::vector<PhysicalRegion>& regions,
                                                 Context ctx,
                                                 Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  // We don't need the axis but we sill need to unpack it
  derez.unpack_dimension();
  const int collapse_dim = derez.unpack_dimension();
  const int init_dim     = derez.unpack_dimension();
  switch (init_dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<uint64_t, 1> out =
        (collapse_dim >= 0) ? derez.unpack_accessor_WO<uint64_t, 1>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_WO<uint64_t, 1>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) out[x] = SumReduction<uint64_t>::identity;
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<uint64_t, 2> out =
        (collapse_dim >= 0) ? derez.unpack_accessor_WO<uint64_t, 2>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_WO<uint64_t, 2>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          out[x][y] = SumReduction<uint64_t>::identity;
      break;
    }
    default: assert(false);  // shouldn't see any other cases
  }
  const int dim = derez.unpack_dimension();
  switch (dim) {
    // Should never get the case of 1 as this would just be a copy since
    // reducing our only dimension should have called CountNonzeroReducTask
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRW<uint64_t, 2> inout =
        (collapse_dim >= 0) ? derez.unpack_accessor_RW<uint64_t, 2, 1>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_RW<uint64_t, 2>(regions[0], rect);
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          SumReduction<uint64_t>::template fold<true /*exclusive*/>(
            inout[x][y], static_cast<uint64_t>(in[x][y] != T{0}));
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRW<uint64_t, 3> inout =
        (collapse_dim >= 0) ? derez.unpack_accessor_RW<uint64_t, 3, 2>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_RW<uint64_t, 3>(regions[0], rect);
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
            SumReduction<uint64_t>::template fold<true /*exclusive*/>(
              inout[x][y][z], static_cast<uint64_t>(in[x][y][z] != T{0}));
      break;
    }
    default: assert(false);
  }
}

#ifdef LEGATE_USE_OPENMP
template <typename T>
/*static*/ void CountNonzeroTask<T>::omp_variant(const Task* task,
                                                 const std::vector<PhysicalRegion>& regions,
                                                 Context ctx,
                                                 Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int axis         = derez.unpack_dimension();
  const int collapse_dim = derez.unpack_dimension();
  const int init_dim     = derez.unpack_dimension();
  switch (init_dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<uint64_t, 1> out =
        (collapse_dim >= 0) ? derez.unpack_accessor_WO<uint64_t, 1>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_WO<uint64_t, 1>(regions[0], rect);
#pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) out[x] = SumReduction<uint64_t>::identity;
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<uint64_t, 2> out =
        (collapse_dim >= 0) ? derez.unpack_accessor_WO<uint64_t, 2>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_WO<uint64_t, 2>(regions[0], rect);
#pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          out[x][y] = SumReduction<uint64_t>::identity;
      break;
    }
    default: assert(false);  // shouldn't see any other cases
  }
  const int dim = derez.unpack_dimension();
  switch (dim) {
    // Should never get the case of 1 as this would just be a copy since
    // reducing our only dimension should have called CountNonzeroReducTask
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRW<uint64_t, 2> inout =
        (collapse_dim >= 0) ? derez.unpack_accessor_RW<uint64_t, 2, 1>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_RW<uint64_t, 2>(regions[0], rect);
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      if (axis == 0) {
        // Flip dimension order to avoid races between threads on the collapsing dim
#pragma omp parallel for
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            SumReduction<uint64_t>::template fold<true /*exclusive*/>(
              inout[x][y], static_cast<uint64_t>(in[x][y] != (T)0));
      } else {
#pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            SumReduction<uint64_t>::template fold<true /*exclusive*/>(
              inout[x][y], static_cast<uint64_t>(in[x][y] != (T)0));
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRW<uint64_t, 3> inout =
        (collapse_dim >= 0) ? derez.unpack_accessor_RW<uint64_t, 3, 2>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_RW<uint64_t, 3>(regions[0], rect);
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      if (axis == 0) {
        // Flip dimension order to avoid races between threads on the collapsing dim
#pragma omp parallel for
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              SumReduction<uint64_t>::template fold<true /*exclusive*/>(
                inout[x][y][z], static_cast<uint64_t>(in[x][y][z] != (T)0));
      } else {
#pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              SumReduction<uint64_t>::template fold<true /*exclusive*/>(
                inout[x][y][z], static_cast<uint64_t>(in[x][y][z] != (T)0));
      }
      break;
    }
    default: assert(false);
  }
}
#endif

namespace detail {
template <typename T>
/*static*/ uint64_t count_nonzero_reduc_helper(const Task* task,
                                               const std::vector<PhysicalRegion>& regions,
                                               LegateDeserializer& derez)
{
  const int dim   = derez.unpack_dimension();
  uint64_t result = SumReduction<uint64_t>::identity;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) return result;
      const AccessorRO<T, 1> in = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
        if (in[x] != (T)0) result++;
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return result;
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          if (in[x][y] != (T)0) result++;
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) return result;
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
            if (in[x][y][z] != (T)0) result++;
      break;
    }
    default: assert(false);
  }
  return result;
}
}  // namespace detail

template <typename T>
/*static*/ uint64_t CountNonzeroReducTask<T>::cpu_variant(
  const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  return detail::count_nonzero_reduc_helper<T>(task, regions, derez);
}

template <typename T>
/*static*/ void CountNonzeroReducWriteTask<T>::cpu_variant(
  const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  auto result   = detail::count_nonzero_reduc_helper<T>(task, regions, derez);
  const int dim = derez.unpack_dimension();
  if (dim == 1) {
    const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
    assert(rect.volume() == 1);
    const AccessorWO<uint64_t, 1> out = derez.unpack_accessor_WO<uint64_t, 1>(regions[1], rect);
    out[rect.lo]                      = result;
  } else if (dim == 2) {
    const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
    assert(rect.volume() == 1);
    const AccessorWO<uint64_t, 2> out = derez.unpack_accessor_WO<uint64_t, 2>(regions[1], rect);
    out[rect.lo]                      = result;
  } else if (dim == 3) {
    const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
    assert(rect.volume() == 1);
    const AccessorWO<uint64_t, 3> out = derez.unpack_accessor_WO<uint64_t, 3>(regions[1], rect);
    out[rect.lo]                      = result;
  }
}

#ifdef LEGATE_USE_OPENMP
namespace detail {
template <typename T>
/*static*/ uint64_t count_nonzero_reduc_omp_helper(const Task* task,
                                                   const std::vector<PhysicalRegion>& regions,
                                                   LegateDeserializer& derez)
{
  const int dim         = derez.unpack_dimension();
  const int max_threads = omp_get_max_threads();
  uint64_t* results     = static_cast<uint64_t*>(alloca(max_threads * sizeof(uint64_t)));
  for (int i = 0; i < max_threads; i++) results[i] = SumReduction<uint64_t>::identity;
  switch (dim) {
    case 1: {
      const Rect<1> rect        = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      const AccessorRO<T, 1> in = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
      if (rect.empty()) return results[0];
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for nowait
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          SumReduction<uint64_t>::template fold<true /*exclusive*/>(results[tid], in[x] != (T)0);
      }
      break;
    }
    case 2: {
      const Rect<2> rect        = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[0], rect);
      if (rect.empty()) return results[0];
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for nowait
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            SumReduction<uint64_t>::template fold<true /*exclusive*/>(results[tid],
                                                                      in[x][y] != (T)0);
      }
      break;
    }
    case 3: {
      const Rect<3> rect        = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[0], rect);
      if (rect.empty()) return results[0];
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for nowait
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              SumReduction<uint64_t>::template fold<true /*exclusive*/>(results[tid],
                                                                        in[x][y][z] != (T)0);
      }
      break;
    }
    default: assert(false);
  }
  uint64_t result = results[0];
  for (int i = 1; i < max_threads; i++)
    SumReduction<uint64_t>::template fold<true /*exclusive*/>(result, results[i]);
  return result;
}
}  // namespace detail

template <typename T>
/*static*/ uint64_t CountNonzeroReducTask<T>::omp_variant(
  const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  return detail::count_nonzero_reduc_omp_helper<T>(task, regions, derez);
}

template <typename T>
/*static*/ void CountNonzeroReducWriteTask<T>::omp_variant(
  const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  auto result   = detail::count_nonzero_reduc_omp_helper<T>(task, regions, derez);
  const int dim = derez.unpack_dimension();
  if (dim == 1) {
    const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
    assert(rect.volume() == 1);
    const AccessorWO<uint64_t, 1> out = derez.unpack_accessor_WO<uint64_t, 1>(regions[1], rect);
    out[rect.lo]                      = result;
  } else if (dim == 2) {
    const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
    assert(rect.volume() == 1);
    const AccessorWO<uint64_t, 2> out = derez.unpack_accessor_WO<uint64_t, 2>(regions[1], rect);
    out[rect.lo]                      = result;
  } else if (dim == 3) {
    const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
    assert(rect.volume() == 1);
    const AccessorWO<uint64_t, 3> out = derez.unpack_accessor_WO<uint64_t, 3>(regions[1], rect);
    out[rect.lo]                      = result;
  }
}
#endif

template <typename T>
/*static*/ void NonzeroTask<T>::cpu_variant(const Task* task,
                                            const std::vector<PhysicalRegion>& regions,
                                            Context ctx,
                                            Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int in_dim = derez.unpack_dimension();
  assert(in_dim > 0);
  switch (in_dim) {
    case 1: {
      const Rect<1> in_rect     = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      const AccessorRO<T, 1> in = derez.unpack_accessor_RO<T, 1>(regions[0], in_rect);
      const Rect<2> out_rect    = regions[1];
      const AccessorRW<uint64_t, 2> out =
        derez.unpack_accessor_RW<uint64_t, 2>(regions[1], out_rect);
      coord_t current_out = out_rect.lo[1];
      for (coord_t x = in_rect.lo[0]; x <= in_rect.hi[0]; x++) {
        if (in[x] != (T)0) { out[out_rect.lo[0]][current_out++] = static_cast<uint64_t>(x); }
      }
      break;
    }
    case 2: {
      const Rect<2> in_rect     = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[0], in_rect);
      const Rect<2> out_rect    = regions[1];
      const AccessorRW<uint64_t, 2> out =
        derez.unpack_accessor_RW<uint64_t, 2>(regions[1], out_rect);
      coord_t current_out = out_rect.lo[1];
      for (coord_t x = in_rect.lo[0]; x <= in_rect.hi[0]; x++) {
        for (coord_t y = in_rect.lo[1]; y <= in_rect.hi[1]; y++) {
          if (in[x][y] != (T)0) {
            out[out_rect.lo[0]][current_out]     = static_cast<uint64_t>(x);
            out[out_rect.lo[0] + 1][current_out] = static_cast<uint64_t>(y);
            ++current_out;
          }
        }
      }
      break;
    }
    case 3: {
      const Rect<3> in_rect     = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[0], in_rect);
      const Rect<2> out_rect    = regions[1];
      const AccessorRW<uint64_t, 2> out =
        derez.unpack_accessor_RW<uint64_t, 2>(regions[1], out_rect);
      coord_t current_out = out_rect.lo[1];
      for (coord_t x = in_rect.lo[0]; x <= in_rect.hi[0]; x++) {
        for (coord_t y = in_rect.lo[1]; y <= in_rect.hi[1]; y++) {
          for (coord_t z = in_rect.lo[2]; z <= in_rect.hi[2]; z++) {
            if (in[x][y][z] != (T)0) {
              out[out_rect.lo[0]][current_out]     = static_cast<uint64_t>(x);
              out[out_rect.lo[0] + 1][current_out] = static_cast<uint64_t>(y);
              out[out_rect.lo[0] + 2][current_out] = static_cast<uint64_t>(z);
              ++current_out;
            }
          }
        }
      }
      break;
    }
    default: assert(false);
  }
  return;
}

#ifdef LEGATE_USE_OPENMP
/*
This implementation has every thread check an equal range of elements for nonzeros and then put them
into temporary vectors.  After that, a partial sum is computed, giving offsets for every partial
vector into the final solution.  Finally, every thread copies its nonzeros into the final vector
using the offset computed with the partial sum.

This implementation is potentially susceptible to load imbalance in the final copy (e.g., all
nonzeros in one thread).
 */
template <typename T>
/*static*/ void NonzeroTask<T>::omp_variant(const Task* task,
                                            const std::vector<PhysicalRegion>& regions,
                                            Context ctx,
                                            Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int in_dim = derez.unpack_dimension();
  assert(in_dim > 0);
  switch (in_dim) {
    case 1: {
      using result_t            = std::vector<std::vector<uint64_t>>;
      const Rect<1> in_rect     = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      const AccessorRO<T, 1> in = derez.unpack_accessor_RO<T, 1>(regions[0], in_rect);
      const Rect<2> out_rect    = regions[1];
      const AccessorRW<uint64_t, 2> out =
        derez.unpack_accessor_RW<uint64_t, 2>(regions[1], out_rect);
      const int max_threads = omp_get_max_threads();
      result_t result(max_threads);
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for nowait
        for (coord_t x = in_rect.lo[0]; x <= in_rect.hi[0]; x++) {
          if (in[x] != (T)0) { result[tid].push_back(static_cast<uint64_t>(x)); }
        }
      }
      // For computing a prefix sum of sizes.  Leave a space for the leading zero.
      std::vector<size_t> sizes(max_threads + 1);
      // Get sizes of the result vectors
      std::transform(result.begin(), result.end(), sizes.begin() + 1, [](const auto& vec) {
        return vec.size();
      });
      // prefix sum of sizes
      std::partial_sum(sizes.begin() + 1, sizes.end(), sizes.begin() + 1);
#pragma omp parallel
      {
        const int tid       = omp_get_thread_num();
        coord_t current_out = out_rect.lo[1] + sizes[tid];
        for (auto it = result[tid].begin(); it < result[tid].end(); ++it) {
          out[out_rect.lo[0]][current_out++] = *it;
        }
      }
      break;
    }
    case 2: {
      using result_t            = std::vector<std::vector<std::tuple<uint64_t, uint64_t>>>;
      const Rect<2> in_rect     = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[0], in_rect);
      const Rect<2> out_rect    = regions[1];
      const AccessorRW<uint64_t, 2> out =
        derez.unpack_accessor_RW<uint64_t, 2>(regions[1], out_rect);
      const int max_threads = omp_get_max_threads();
      result_t result(max_threads);
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for nowait collapse(2)
        for (coord_t x = in_rect.lo[0]; x <= in_rect.hi[0]; x++) {
          for (coord_t y = in_rect.lo[1]; y <= in_rect.hi[1]; y++) {
            if (in[x][y] != (T)0) { result[tid].push_back({x, y}); }
          }
        }
      }
      // For computing a prefix sum of sizes.  Leave a space for the leading zero.
      std::vector<size_t> sizes(max_threads + 1);
      // Get sizes of the result vectors
      std::transform(result.begin(), result.end(), sizes.begin() + 1, [](const auto& vec) {
        return vec.size();
      });
      // prefix sum of sizes
      std::partial_sum(sizes.begin() + 1, sizes.end(), sizes.begin() + 1);
#pragma omp parallel
      {
        const int tid       = omp_get_thread_num();
        coord_t current_out = out_rect.lo[1] + sizes[tid];
        for (auto it = result[tid].begin(); it < result[tid].end(); ++it) {
          out[out_rect.lo[0]][current_out]       = static_cast<uint64_t>(std::get<0>(*it));
          out[out_rect.lo[0] + 1][current_out++] = static_cast<uint64_t>(std::get<1>(*it));
        }
      }
      break;
    }
    case 3: {
      using result_t        = std::vector<std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>>;
      const Rect<3> in_rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[0], in_rect);
      const Rect<2> out_rect    = regions[1];
      const AccessorRW<uint64_t, 2> out =
        derez.unpack_accessor_RW<uint64_t, 2>(regions[1], out_rect);
      const int max_threads = omp_get_max_threads();
      result_t result(max_threads);
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for nowait collapse(2)
        for (coord_t x = in_rect.lo[0]; x <= in_rect.hi[0]; x++) {
          for (coord_t y = in_rect.lo[1]; y <= in_rect.hi[1]; y++) {
            for (coord_t z = in_rect.lo[2]; z <= in_rect.hi[2]; z++) {
              if (in[x][y][z] != (T)0) { result[tid].push_back({x, y, z}); }
            }
          }
        }
      }
      // For computing a prefix sum of sizes.  Leave a space for the leading zero.
      std::vector<size_t> sizes(max_threads + 1);
      // Get sizes of the result vectors
      std::transform(result.begin(), result.end(), sizes.begin() + 1, [](const auto& vec) {
        return vec.size();
      });
      // prefix sum of sizes
      std::partial_sum(sizes.begin() + 1, sizes.end(), sizes.begin() + 1);
#pragma omp parallel
      {
        const int tid       = omp_get_thread_num();
        coord_t current_out = out_rect.lo[1] + sizes[tid];
        for (auto it = result[tid].begin(); it < result[tid].end(); ++it) {
          out[out_rect.lo[0]][current_out]       = static_cast<uint64_t>(std::get<0>(*it));
          out[out_rect.lo[0] + 1][current_out]   = static_cast<uint64_t>(std::get<1>(*it));
          out[out_rect.lo[0] + 2][current_out++] = static_cast<uint64_t>(std::get<2>(*it));
        }
      }
      break;
    }
    default: assert(false);
  }
  return;
}
#endif

template <typename T>
void ConvertRangeToRectTask<T>::cpu_variant(const Task* task,
                                            const std::vector<PhysicalRegion>& regions,
                                            Context ctx,
                                            Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const coord_t nonzero_dim = derez.unpack_32bit_int();
  const int dim             = derez.unpack_dimension();
  assert(dim == 1);
  const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  assert(!rect.empty());
  const auto begin                 = rect.lo[0];
  const auto end                   = rect.hi[0];
  const AccessorRO<T, 1> in        = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
  const AccessorRW<Rect<2>, 1> out = derez.unpack_accessor_RW<Rect<2>, 1>(regions[1], rect);

  auto ptr       = out.ptr(rect);
  coord_t pt1[2] = {0, 0};
  coord_t pt2[2] = {nonzero_dim, static_cast<coord_t>(in[rect.lo[0]] - 1)};
  new (ptr) Rect<2>(Point<2>(pt1), Point<2>(pt2));
  for (coord_t x = begin + 1; x <= end; x++) {
    coord_t pt1[2] = {0, static_cast<coord_t>(in[x - 1])};
    coord_t pt2[2] = {nonzero_dim, static_cast<coord_t>(in[x] - 1)};
    new (ptr + x) Rect<2>(Point<2>(pt1), Point<2>(pt2));
  }
};

#ifdef LEGATE_USE_OPENMP
template <typename T>
void ConvertRangeToRectTask<T>::omp_variant(const Task* task,
                                            const std::vector<PhysicalRegion>& regions,
                                            Context ctx,
                                            Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const coord_t nonzero_dim = derez.unpack_32bit_int();
  const int dim             = derez.unpack_dimension();
  assert(dim == 1);
  const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  assert(!rect.empty());
  const auto begin                 = rect.lo[0];
  const auto end                   = rect.hi[0];
  const AccessorRO<T, 1> in        = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
  const AccessorRW<Rect<2>, 1> out = derez.unpack_accessor_RW<Rect<2>, 1>(regions[1], rect);

  auto ptr       = out.ptr(rect);
  coord_t pt1[2] = {0, 0};
  coord_t pt2[2] = {nonzero_dim, static_cast<coord_t>(in[rect.lo[0]] - 1)};
  new (ptr) Rect<2>(Point<2>(pt1), Point<2>(pt2));
#pragma omp parallel for
  for (coord_t x = begin + 1; x <= end; x++) {
    coord_t pt1[2] = {0, static_cast<coord_t>(in[x - 1])};
    coord_t pt2[2] = {nonzero_dim, static_cast<coord_t>(in[x] - 1)};
    new (ptr + x) Rect<2>(Point<2>(pt1), Point<2>(pt2));
  }
};
#endif  // LEGATE_USE_OPENMP

INSTANTIATE_ALL_TASKS(CountNonzeroTask,
                      static_cast<int>(NumPyOpCode::NUMPY_COUNT_NONZERO) * NUMPY_TYPE_OFFSET)
INSTANTIATE_ALL_TASKS(CountNonzeroReducTask,
                      static_cast<int>(NumPyOpCode::NUMPY_COUNT_NONZERO) * NUMPY_TYPE_OFFSET +
                        NUMPY_REDUCTION_VARIANT_OFFSET)
INSTANTIATE_ALL_TASKS(CountNonzeroReducWriteTask,
                      static_cast<int>(NumPyOpCode::NUMPY_COUNT_NONZERO_REDUC) * NUMPY_TYPE_OFFSET)
INSTANTIATE_ALL_TASKS(NonzeroTask, static_cast<int>(NumPyOpCode::NUMPY_NONZERO) * NUMPY_TYPE_OFFSET)
INSTANTIATE_INT_TASKS(ConvertRangeToRectTask,
                      static_cast<int>(NumPyOpCode::NUMPY_CONVERT_TO_RECT) * NUMPY_TYPE_OFFSET)
INSTANTIATE_UINT_TASKS(ConvertRangeToRectTask,
                       static_cast<int>(NumPyOpCode::NUMPY_CONVERT_TO_RECT) * NUMPY_TYPE_OFFSET)

}  // namespace numpy
}  // namespace legate

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  REGISTER_ALL_TASKS(legate::numpy::CountNonzeroTask)
  REGISTER_ALL_TASKS(legate::numpy::CountNonzeroReducWriteTask)
  REGISTER_ALL_TASKS_WITH_REDUCTION_ARG_RETURN(legate::numpy::CountNonzeroReducTask, SumReduction)
  REGISTER_ALL_TASKS(legate::numpy::NonzeroTask)
  REGISTER_INT_TASKS(legate::numpy::ConvertRangeToRectTask)
  REGISTER_UINT_TASKS(legate::numpy::ConvertRangeToRectTask)
}
}  // namespace
