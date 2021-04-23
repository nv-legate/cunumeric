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

#include "norm.h"
#include "proj.h"
#ifdef LEGATE_USE_OPENMP
#include <alloca.h>
#include <omp.h>
#endif

using namespace Legion;

namespace legate {
namespace numpy {

template <typename T>
/*static*/ void NormTask<T>::cpu_variant(const Task* task,
                                         const std::vector<PhysicalRegion>& regions,
                                         Context ctx,
                                         Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  // Still need to unpack the axis
  derez.unpack_dimension();
  const int collapse_dim = derez.unpack_dimension();
  const int init_dim     = derez.unpack_dimension();
  switch (init_dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<T, 1> out =
        (collapse_dim >= 0) ? derez.unpack_accessor_WO<T, 1>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_WO<T, 1>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) out[x] = SumReduction<T>::identity;
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<T, 2> out =
        (collapse_dim >= 0) ? derez.unpack_accessor_WO<T, 2>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_WO<T, 2>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) out[x][y] = SumReduction<T>::identity;
      break;
    }
    default: assert(false);  // shouldn't see any other cases
  }
  const int dim   = derez.unpack_dimension();
  const int order = task->futures[0].get_result<int>();
  assert(order > 0);
  switch (dim) {
    // Should never get the case of 1 as this would just be a copy since
    // reducing our only dimension should have called SumReducTask
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorRW<T, 2> inout =
        (collapse_dim >= 0) ? derez.unpack_accessor_RW<T, 2, 1>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_RW<T, 2>(regions[0], rect);
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      switch (order) {
        case 1: {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
              T val = in[x][y];
              if (val < T{0}) val = -val;
              SumReduction<T>::template fold<true /*exclusive*/>(inout[x][y], val);
            }
          break;
        }
        case 2: {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
              T val = in[x][y];
              ProdReduction<T>::template fold<true /*exclusive*/>(val, val);
              SumReduction<T>::template fold<true /*exclusive*/>(inout[x][y], val);
            }
          break;
        }
        default: {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
              T val = in[x][y];
              if (val < T{0}) val = -val;
              T prod = val;
              for (int i = 0; i < (order - 1); i++)
                ProdReduction<T>::template fold<true /*exclusive*/>(prod, val);
              SumReduction<T>::template fold<true /*exclusive*/>(inout[x][y], prod);
            }
          break;
        }
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) return;
      const AccessorRW<T, 3> inout =
        (collapse_dim >= 0) ? derez.unpack_accessor_RW<T, 3, 2>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_RW<T, 3>(regions[0], rect);
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      switch (order) {
        case 1: {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
                T val = in[x][y][z];
                if (val < T{0}) val = -val;
                SumReduction<T>::template fold<true /*exclusive*/>(inout[x][y][z], val);
              }
          break;
        }
        case 2: {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
                T val = in[x][y][z];
                ProdReduction<T>::template fold<true /*exclusive*/>(val, val);
                SumReduction<T>::template fold<true /*exclusive*/>(inout[x][y][z], val);
              }
          break;
        }
        default: {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
                T val = in[x][y][z];
                if (val < T{0}) val = -val;
                T prod = val;
                for (int i = 0; i < (order - 1); i++)
                  ProdReduction<T>::template fold<true /*exclusive*/>(prod, val);
                SumReduction<T>::template fold<true /*exclusive*/>(inout[x][y][z], prod);
              }
          break;
        }
      }
      break;
    }
    default: assert(false);
  }
}

#ifdef LEGATE_USE_OPENMP
template <typename T>
/*static*/ void NormTask<T>::omp_variant(const Task* task,
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
      const AccessorWO<T, 1> out =
        (collapse_dim >= 0) ? derez.unpack_accessor_WO<T, 1>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_WO<T, 1>(regions[0], rect);
#pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) out[x] = SumReduction<T>::identity;
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<T, 2> out =
        (collapse_dim >= 0) ? derez.unpack_accessor_WO<T, 2>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_WO<T, 2>(regions[0], rect);
#pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) out[x][y] = SumReduction<T>::identity;
      break;
    }
    default: assert(false);  // shouldn't see any other cases
  }
  const int dim   = derez.unpack_dimension();
  const int order = task->futures[0].get_result<int>();
  assert(order > 0);
  switch (dim) {
    // Should never get the case of 1 as this would just be a copy since
    // reducing our only dimension should have called SumReducTask
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorRW<T, 2> inout =
        (collapse_dim >= 0) ? derez.unpack_accessor_RW<T, 2, 1>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_RW<T, 2>(regions[0], rect);
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      switch (order) {
        case 1: {
          if (axis == 0) {
// Flip dimension order to avoid races between threads on the collapsing dim
#pragma omp parallel for
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
                T val = in[x][y];
                if (val < T{0}) val = -val;
                SumReduction<T>::template fold<true /*exclusive*/>(inout[x][y], val);
              }
          } else {
#pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
                T val = in[x][y];
                if (val < T{0}) val = -val;
                SumReduction<T>::template fold<true /*exclusive*/>(inout[x][y], val);
              }
          }
          break;
        }
        case 2: {
          if (axis == 0) {
// Flip dimension order to avoid races between threads on the collapsing dim
#pragma omp parallel for
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
                T val = in[x][y];
                ProdReduction<T>::template fold<true /*exclusive*/>(val, val);
                SumReduction<T>::template fold<true /*exclusive*/>(inout[x][y], val);
              }
          } else {
#pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
                T val = in[x][y];
                ProdReduction<T>::template fold<true /*exclusive*/>(val, val);
                SumReduction<T>::template fold<true /*exclusive*/>(inout[x][y], val);
              }
          }
          break;
        }
        default: {
          if (axis == 0) {
// Flip dimension order to avoid races between threads on the collapsing dim
#pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
                T val = in[x][y];
                if (val < T{0}) val = -val;
                T prod = val;
                for (int i = 0; i < (order - 1); i++)
                  ProdReduction<T>::template fold<true /*exclusive*/>(prod, val);
                SumReduction<T>::template fold<true /*exclusive*/>(inout[x][y], prod);
              }
          } else {
#pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
                T val = in[x][y];
                if (val < T{0}) val = -val;
                T prod = val;
                for (int i = 0; i < (order - 1); i++)
                  ProdReduction<T>::template fold<true /*exclusive*/>(prod, val);
                SumReduction<T>::template fold<true /*exclusive*/>(inout[x][y], prod);
              }
          }
          break;
        }
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) return;
      const AccessorRW<T, 3> inout =
        (collapse_dim >= 0) ? derez.unpack_accessor_RW<T, 3, 2>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_RW<T, 3>(regions[0], rect);
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      switch (order) {
        case 1: {
          if (axis == 0) {
// Flip dimension order to avoid races between threads on the collapsing dim
#pragma omp parallel for
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
                  T val = in[x][y][z];
                  if (val < T{0}) val = -val;
                  SumReduction<T>::template fold<true /*exclusive*/>(inout[x][y][z], val);
                }
          } else {
#pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
                  T val = in[x][y][z];
                  if (val < T{0}) val = -val;
                  SumReduction<T>::template fold<true /*exclusive*/>(inout[x][y][z], val);
                }
          }
          break;
        }
        case 2: {
          if (axis == 0) {
// Flip dimension order to avoid races between threads on the collapsing dim
#pragma omp parallel for
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
                  T val = in[x][y][z];
                  ProdReduction<T>::template fold<true /*excluisve*/>(val, val);
                  SumReduction<T>::template fold<true /*exclusive*/>(inout[x][y][z], val);
                }
          } else {
#pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
                  T val = in[x][y][z];
                  ProdReduction<T>::template fold<true /*exclusive*/>(val, val);
                  SumReduction<T>::template fold<true /*exclusive*/>(inout[x][y][z], val);
                }
          }
          break;
        }
        default: {
          if (axis == 0) {
// Flip dimension order to avoid races between threads on the collapsing dim
#pragma omp parallel for
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
                  T val = in[x][y][z];
                  if (val < T{0}) val = -val;
                  T prod = val;
                  for (int i = 0; i < (order - 1); i++)
                    ProdReduction<T>::template fold<true /*exclusive*/>(prod, val);
                  SumReduction<T>::template fold<true /*exclusive*/>(inout[x][y][z], prod);
                }
          } else {
#pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
                  T val = in[x][y][z];
                  if (val < T{0}) val = -val;
                  T prod = val;
                  for (int i = 0; i < (order - 1); i++)
                    ProdReduction<T>::template fold<true /*exclusive*/>(prod, val);
                  SumReduction<T>::template fold<true /*exclusive*/>(inout[x][y][z], prod);
                }
          }
          break;
        }
      }
      break;
    }
    default: assert(false);
  }
}
#endif

template <typename T>
/*static*/ T NormReducTask<T>::cpu_variant(const Task* task,
                                           const std::vector<PhysicalRegion>& regions,
                                           Context ctx,
                                           Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int dim   = derez.unpack_dimension();
  T result        = SumReduction<T>::identity;
  const int order = task->futures[0].get_result<int>();
  assert(order > 0);
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
      switch (order) {
        case 1: {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
            T val = in[x];
            if (val < T{0}) val = -val;
            SumReduction<T>::template fold<true /*exclusive*/>(result, val);
          }
          break;
        }
        case 2: {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
            T val = in[x];
            ProdReduction<T>::template fold<true /*exclusive*/>(val, val);
            SumReduction<T>::template fold<true /*exclusive*/>(result, val);
          }
          break;
        }
        default: {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
            T val = in[x];
            if (val < T{0}) val = -val;
            T prod = val;
            for (int i = 0; i < (order - 1); i++)
              ProdReduction<T>::template fold<true /*exclusive*/>(prod, val);
            SumReduction<T>::template fold<true /*exclusive*/>(result, prod);
          }
          break;
        }
      }
      break;
    }
    default: assert(false);  // should have any other dimensions
  }
  return result;
}

#ifdef LEGATE_USE_OPENMP
template <typename T>
/*static*/ T NormReducTask<T>::omp_variant(const Task* task,
                                           const std::vector<PhysicalRegion>& regions,
                                           Context ctx,
                                           Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int dim         = derez.unpack_dimension();
  const int max_threads = omp_get_max_threads();
  T* results            = (T*)alloca(max_threads * sizeof(T));
  for (int i = 0; i < max_threads; i++) results[i] = SumReduction<T>::identity;
  const int order = task->futures[0].get_result<int>();
  assert(order > 0);
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
      switch (order) {
        case 1: {
#pragma omp parallel
          {
            const int tid = omp_get_thread_num();
#pragma omp for nowait
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
              T val = in[x];
              if (val < T{0}) val = -val;
              SumReduction<T>::template fold<true /*exclusive*/>(results[tid], val);
            }
          }
          break;
        }
        case 2: {
#pragma omp parallel
          {
            const int tid = omp_get_thread_num();
#pragma omp for nowait
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
              T val = in[x];
              ProdReduction<T>::template fold<true /*exclusive*/>(val, val);
              SumReduction<T>::template fold<true /*exclusive*/>(results[tid], val);
            }
          }
          break;
        }
        default: {
#pragma omp parallel
          {
            const int tid = omp_get_thread_num();
#pragma omp for nowait
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
              T val = in[x];
              if (val < T{0}) val = -val;
              T prod = val;
              for (int i = 0; i < (order - 1); i++)
                ProdReduction<T>::template fold<true /*exclusive*/>(prod, val);
              SumReduction<T>::template fold<true /*exclusive*/>(results[tid], prod);
            }
          }
          break;
        }
      }
      break;
    }
    default: assert(false);  // should have any other dimensions
  }
  T result = results[0];
  for (int i = 1; i < max_threads; i++)
    SumReduction<T>::template fold<true /*exclusive*/>(result, results[i]);
  return result;
}
#endif  // LEGATE_USE_OPENMP

INSTANTIATE_ALL_TASKS(NormTask, static_cast<int>(NumPyOpCode::NUMPY_NORM) * NUMPY_TYPE_OFFSET)
INSTANTIATE_ALL_TASKS(NormReducTask,
                      static_cast<int>(NumPyOpCode::NUMPY_NORM) * NUMPY_TYPE_OFFSET +
                        NUMPY_REDUCTION_VARIANT_OFFSET)

}  // namespace numpy
}  // namespace legate

namespace  // unnammed
{
static void __attribute__((constructor)) register_tasks(void)
{
  REGISTER_ALL_TASKS(legate::numpy::NormTask)
  REGISTER_ALL_TASKS_WITH_REDUCTION_RETURN(legate::numpy::NormReducTask, SumReduction)
}
}  // namespace
