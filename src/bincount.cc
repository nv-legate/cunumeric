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
#include "proj.h"
#ifdef LEGATE_USE_OPENMP
#  include <alloca.h>
#  include <omp.h>
#endif

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
/*static*/ void BinCountTask<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                             Runtime* runtime) {
  LegateDeserializer            derez(task->args, task->arglen);
  const int                     collapse_dim   = derez.unpack_dimension();
  const int                     collapse_index = derez.unpack_dimension();
  const Rect<1>                 bin_rect       = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  const AccessorWO<uint64_t, 1> out =
      (collapse_dim >= 0)
          ? derez.unpack_accessor_WO<uint64_t, 1>(regions[0], bin_rect, collapse_dim, task->index_point[collapse_index])
          : derez.unpack_accessor_WO<uint64_t, 1>(regions[0], bin_rect);
  // Initialize all the counts to zero
  for (coord_t x = bin_rect.lo[0]; x <= bin_rect.hi[0]; x++)
    out[x] = SumReduction<uint64_t>::identity;
  const int dim = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
      // Then count all the entries
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
        const coord_t bin = in[x];
        assert(bin_rect.contains(bin));
        SumReduction<uint64_t>::fold<true /*exclusive*/>(out[bin], 1);
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      // Then count all the entries
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
          const coord_t bin = in[x][y];
          assert(bin_rect.contains(bin));
          SumReduction<uint64_t>::fold<true /*exclusive*/>(out[bin], 1);
        }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      // Then count all the entries
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
            const coord_t bin = in[x][y][z];
            assert(bin_rect.contains(bin));
            SumReduction<uint64_t>::fold<true /*exclusive*/>(out[bin], 1);
          }
      break;
    }
    default:
      assert(false);
  }
}

#ifdef LEGATE_USE_OPENMP
template<typename T>
/*static*/ void BinCountTask<T>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                             Runtime* runtime) {
  LegateDeserializer            derez(task->args, task->arglen);
  const int                     collapse_dim   = derez.unpack_dimension();
  const int                     collapse_index = derez.unpack_dimension();
  const Rect<1>                 bin_rect       = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  const AccessorWO<uint64_t, 1> out =
      (collapse_dim >= 0)
          ? derez.unpack_accessor_WO<uint64_t, 1>(regions[0], bin_rect, collapse_dim, task->index_point[collapse_index])
          : derez.unpack_accessor_WO<uint64_t, 1>(regions[0], bin_rect);
// Initialize all the counts to zero
#  pragma omp parallel for
  for (coord_t x = bin_rect.lo[0]; x <= bin_rect.hi[0]; x++)
    out[x] = SumReduction<uint64_t>::identity;
  const int    dim         = derez.unpack_dimension();
  const int    max_threads = omp_get_max_threads();
  const size_t bin_volume  = bin_rect.volume();
  uint64_t*    temp        = (uint64_t*)alloca(max_threads * bin_volume * sizeof(uint64_t));
#  pragma omp  parallel for
  for (int t = 0; t < max_threads; t++)
    for (unsigned b = 0; b < bin_volume; b++)
      temp[t * bin_volume + b] = SumReduction<uint64_t>::identity;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
#  pragma omp parallel
      {
        const int offset = omp_get_thread_num() * bin_volume;
// Then count all the entries
#  pragma omp for nowait
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
          const coord_t bin = in[x];
          assert(bin_rect.contains(bin));
          // Integer reductions are deterministic
          SumReduction<uint64_t>::fold<true /*exclusive*/>(temp[offset + bin], 1);
        }
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
#  pragma omp parallel
      {
        const int offset = omp_get_thread_num() * bin_volume;
// Then count all the entries
#  pragma omp for nowait
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
            const coord_t bin = in[x][y];
            assert(bin_rect.contains(bin));
            // Integer reductions are deterministic
            SumReduction<uint64_t>::fold<true /*exclusive*/>(temp[offset + bin], 1);
          }
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
#  pragma omp parallel
      {
        const int offset = omp_get_thread_num() * bin_volume;
// Then count all the entries
#  pragma omp for nowait
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
              const coord_t bin = in[x][y][z];
              assert(bin_rect.contains(bin));
              // Integer reductions are deterministic
              SumReduction<uint64_t>::fold<true /*exclusive*/>(temp[offset + bin], 1);
            }
      }
      break;
    }
    default:
      assert(false);
  }
#  pragma omp parallel for
  for (int t = 0; t < max_threads; t++)
    for (unsigned b = 0; b < bin_volume; b++) {
      const uint64_t count = temp[t * bin_volume + b];
      if (count != SumReduction<uint64_t>::identity) SumReduction<uint64_t>::fold<false /*exclusive*/>(out[b], count);
    }
}
#endif    // LEGATE_USE_OPENMP

template<typename T, typename WT>
/*static*/ void WeightedBinCountTask<T, WT>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                         Runtime* runtime) {
  LegateDeserializer      derez(task->args, task->arglen);
  const int               collapse_dim   = derez.unpack_dimension();
  const int               collapse_index = derez.unpack_dimension();
  const Rect<1>           bin_rect       = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  const AccessorWO<WT, 1> out =
      (collapse_dim >= 0) ? derez.unpack_accessor_WO<WT, 1>(regions[0], bin_rect, collapse_dim, task->index_point[collapse_index])
                          : derez.unpack_accessor_WO<WT, 1>(regions[0], bin_rect);
  // Initialize all the counts to zero
  for (coord_t x = bin_rect.lo[0]; x <= bin_rect.hi[0]; x++)
    out[x] = SumReduction<WT>::identity;
  const int dim = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1>  in      = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
      const AccessorRO<WT, 1> weights = derez.unpack_accessor_RO<WT, 1>(regions[2], rect);
      // Then count all the entries
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
        const coord_t bin = in[x];
        assert(bin_rect.contains(bin));
        SumReduction<WT>::template fold<true /*exclusive*/>(out[bin], weights[x]);
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2>  in      = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      const AccessorRO<WT, 2> weights = derez.unpack_accessor_RO<WT, 2>(regions[2], rect);
      // Then count all the entries
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
          const coord_t bin = in[x][y];
          assert(bin_rect.contains(bin));
          SumReduction<WT>::template fold<true /*exclusive*/>(out[bin], weights[x][y]);
        }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3>  in      = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      const AccessorRO<WT, 3> weights = derez.unpack_accessor_RO<WT, 3>(regions[2], rect);
      // Then count all the entries
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
            const coord_t bin = in[x][y][z];
            assert(bin_rect.contains(bin));
            SumReduction<WT>::template fold<true /*exclusive*/>(out[bin], weights[x][y][z]);
          }
      break;
    }
    default:
      assert(false);
  }
}

#ifdef LEGATE_USE_OPENMP
template<typename T, typename WT>
/*static*/ void WeightedBinCountTask<T, WT>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                         Runtime* runtime) {
  LegateDeserializer      derez(task->args, task->arglen);
  const int               collapse_dim   = derez.unpack_dimension();
  const int               collapse_index = derez.unpack_dimension();
  const Rect<1>           bin_rect       = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  const AccessorWO<WT, 1> out =
      (collapse_dim >= 0) ? derez.unpack_accessor_WO<WT, 1>(regions[0], bin_rect, collapse_dim, task->index_point[collapse_index])
                          : derez.unpack_accessor_WO<WT, 1>(regions[0], bin_rect);
// Initialize all the counts to zero
#  pragma omp parallel for
  for (coord_t x = bin_rect.lo[0]; x <= bin_rect.hi[0]; x++)
    out[x] = SumReduction<WT>::identity;
  const int    dim         = derez.unpack_dimension();
  const int    max_threads = omp_get_max_threads();
  const size_t bin_volume  = bin_rect.volume();
  WT*          temp        = (WT*)alloca(max_threads * bin_volume * sizeof(WT));
#  pragma omp  parallel for
  for (int t = 0; t < max_threads; t++)
    for (unsigned b = 0; b < bin_volume; b++)
      temp[t * bin_volume + b] = SumReduction<WT>::identity;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1>  in      = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
      const AccessorRO<WT, 1> weights = derez.unpack_accessor_RO<WT, 1>(regions[2], rect);
#  pragma omp parallel
      {
        const int offset = omp_get_thread_num() * bin_volume;
// Then count all the entries
#  pragma omp for nowait
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
          const coord_t bin = in[x];
          assert(bin_rect.contains(bin));
          // Integer reductions are deterministic
          SumReduction<WT>::template fold<true /*exclusive*/>(temp[offset + bin], weights[x]);
        }
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2>  in      = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      const AccessorRO<WT, 2> weights = derez.unpack_accessor_RO<WT, 2>(regions[2], rect);
#  pragma omp parallel
      {
        const int offset = omp_get_thread_num() * bin_volume;
// Then count all the entries
#  pragma omp for nowait
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
            const coord_t bin = in[x][y];
            assert(bin_rect.contains(bin));
            // Integer reductions are deterministic
            SumReduction<WT>::template fold<true /*exclusive*/>(temp[offset + bin], weights[x][y]);
          }
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3>  in      = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      const AccessorRO<WT, 3> weights = derez.unpack_accessor_RO<WT, 3>(regions[2], rect);
#  pragma omp parallel
      {
        const int offset = omp_get_thread_num() * bin_volume;
// Then count all the entries
#  pragma omp for nowait
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
              const coord_t bin = in[x][y][z];
              assert(bin_rect.contains(bin));
              // Integer reductions are deterministic
              SumReduction<WT>::template fold<true /*exclusive*/>(temp[offset + bin], weights[x][y][z]);
            }
      }
      break;
    }
    default:
      assert(false);
  }
#  pragma omp parallel for
  for (int t = 0; t < max_threads; t++)
    for (unsigned b = 0; b < bin_volume; b++) {
      const WT result = temp[t * bin_volume + b];
      if (result != SumReduction<WT>::identity) SumReduction<WT>::template fold<false /*exclusive*/>(out[b], result);
    }
}
#endif    // LEGATE_USE_OPENMP

INSTANTIATE_INT_TASKS(BinCountTask, static_cast<int>(NumPyOpCode::NUMPY_BINCOUNT) * NUMPY_TYPE_OFFSET + NUMPY_NORMAL_VARIANT_OFFSET)
INSTANTIATE_UINT_TASKS(BinCountTask,
                       static_cast<int>(NumPyOpCode::NUMPY_BINCOUNT) * NUMPY_TYPE_OFFSET + NUMPY_NORMAL_VARIANT_OFFSET)

#define INSTANTIATE_WEIGHTED_BINCOUNT_TASKS(task, type, base_id)                     \
  template<>                                                                         \
  const int task<type, __half>::TASK_ID = base_id + HALF_LT* NUMPY_MAX_VARIANTS;     \
  template class task<type, __half>;                                                 \
  template<>                                                                         \
  const int task<type, float>::TASK_ID = base_id + FLOAT_LT* NUMPY_MAX_VARIANTS;     \
  template class task<type, float>;                                                  \
  template<>                                                                         \
  const int task<type, double>::TASK_ID = base_id + DOUBLE_LT* NUMPY_MAX_VARIANTS;   \
  template class task<type, double>;                                                 \
  template<>                                                                         \
  const int task<type, int16_t>::TASK_ID = base_id + INT16_LT* NUMPY_MAX_VARIANTS;   \
  template class task<type, int16_t>;                                                \
  template<>                                                                         \
  const int task<type, int32_t>::TASK_ID = base_id + INT32_LT* NUMPY_MAX_VARIANTS;   \
  template class task<type, int32_t>;                                                \
  template<>                                                                         \
  const int task<type, int64_t>::TASK_ID = base_id + INT64_LT* NUMPY_MAX_VARIANTS;   \
  template class task<type, int64_t>;                                                \
  template<>                                                                         \
  const int task<type, uint16_t>::TASK_ID = base_id + UINT16_LT* NUMPY_MAX_VARIANTS; \
  template class task<type, uint16_t>;                                               \
  template<>                                                                         \
  const int task<type, uint32_t>::TASK_ID = base_id + UINT32_LT* NUMPY_MAX_VARIANTS; \
  template class task<type, uint32_t>;                                               \
  template<>                                                                         \
  const int task<type, uint64_t>::TASK_ID = base_id + UINT64_LT* NUMPY_MAX_VARIANTS; \
  template class task<type, uint64_t>;
// No bools for now

INSTANTIATE_WEIGHTED_BINCOUNT_TASKS(WeightedBinCountTask, int16_t, NUMPY_BINCOUNT_OFFSET + INT16_LT * NUMPY_TYPE_OFFSET)
INSTANTIATE_WEIGHTED_BINCOUNT_TASKS(WeightedBinCountTask, int32_t, NUMPY_BINCOUNT_OFFSET + INT32_LT * NUMPY_TYPE_OFFSET)
INSTANTIATE_WEIGHTED_BINCOUNT_TASKS(WeightedBinCountTask, int64_t, NUMPY_BINCOUNT_OFFSET + INT64_LT * NUMPY_TYPE_OFFSET)
INSTANTIATE_WEIGHTED_BINCOUNT_TASKS(WeightedBinCountTask, uint16_t, NUMPY_BINCOUNT_OFFSET + UINT16_LT * NUMPY_TYPE_OFFSET)
INSTANTIATE_WEIGHTED_BINCOUNT_TASKS(WeightedBinCountTask, uint32_t, NUMPY_BINCOUNT_OFFSET + UINT32_LT * NUMPY_TYPE_OFFSET)
INSTANTIATE_WEIGHTED_BINCOUNT_TASKS(WeightedBinCountTask, uint64_t, NUMPY_BINCOUNT_OFFSET + UINT64_LT * NUMPY_TYPE_OFFSET)

}    // namespace numpy
}    // namespace legate

#define REGISTER_WEIGHTED_BINCOUNT_TASKS(task, type) \
  {                                                  \
    task<type, __half>::register_variants();         \
    task<type, float>::register_variants();          \
    task<type, double>::register_variants();         \
    task<type, int16_t>::register_variants();        \
    task<type, int32_t>::register_variants();        \
    task<type, int64_t>::register_variants();        \
    task<type, uint16_t>::register_variants();       \
    task<type, uint32_t>::register_variants();       \
    task<type, uint64_t>::register_variants();       \
  }
// No bools for now

namespace    // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  REGISTER_INT_TASKS(legate::numpy::BinCountTask)
  REGISTER_UINT_TASKS(legate::numpy::BinCountTask)
  REGISTER_WEIGHTED_BINCOUNT_TASKS(legate::numpy::WeightedBinCountTask, int16_t)
  REGISTER_WEIGHTED_BINCOUNT_TASKS(legate::numpy::WeightedBinCountTask, int32_t)
  REGISTER_WEIGHTED_BINCOUNT_TASKS(legate::numpy::WeightedBinCountTask, int64_t)
  REGISTER_WEIGHTED_BINCOUNT_TASKS(legate::numpy::WeightedBinCountTask, uint16_t)
  REGISTER_WEIGHTED_BINCOUNT_TASKS(legate::numpy::WeightedBinCountTask, uint32_t)
  REGISTER_WEIGHTED_BINCOUNT_TASKS(legate::numpy::WeightedBinCountTask, uint64_t)
}
}    // namespace
