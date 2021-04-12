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

#include "prod.h"
#include "proj.h"
#ifdef LEGATE_USE_OPENMP
#  include <alloca.h>
#  include <omp.h>
#endif

using namespace Legion;

namespace legate {
namespace numpy {

#if 0
    template<typename T> template<typename TASK>
    /*static*/ void ProdTask<T>::set_layout_constraints(LegateVariant variant, 
                                 TaskLayoutConstraintSet &layout_constraints)
    {
      // Need a reduction accessor for the first region
      layout_constraints.add_layout_constraint(0, 
          Legate::get_reduction_layout(ProdReduction<T>::REDOP_ID));
      for (int idx = 1; idx < TASK::REGIONS; idx++)
        layout_constraints.add_layout_constraint(idx, Legate::get_soa_layout());
    }
#endif

template<typename T>
/*static*/ void ProdTask<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                         Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  // Still need to unpack the axis
  derez.unpack_dimension();
  const int collapse_dim  = derez.unpack_dimension();
  const int init_dim      = derez.unpack_dimension();
  const T   initial_value = (task->futures.size() == 1) ? task->futures[0].get_result<T>() : ProdReduction<T>::identity;
  switch (init_dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<T, 1> out =
          (collapse_dim >= 0) ? derez.unpack_accessor_WO<T, 1>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                              : derez.unpack_accessor_WO<T, 1>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        out[x] = initial_value;
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<T, 2> out =
          (collapse_dim >= 0) ? derez.unpack_accessor_WO<T, 2>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                              : derez.unpack_accessor_WO<T, 2>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          out[x][y] = initial_value;
      break;
    }
    default:
      assert(false);    // shouldn't see any other cases
  }
  const int dim = derez.unpack_dimension();
  switch (dim) {
    // Should never get the case of 1 as this would just be a copy since
    // reducing our only dimension should have called ProdReducTask
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorRW<T, 2> inout =
          (collapse_dim >= 0) ? derez.unpack_accessor_RW<T, 2, 1>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                              : derez.unpack_accessor_RW<T, 2>(regions[0], rect);
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          ProdReduction<T>::template fold<true /*exclusive*/>(inout[x][y], in[x][y]);
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) return;
      const AccessorRW<T, 3> inout =
          (collapse_dim >= 0) ? derez.unpack_accessor_RW<T, 3, 2>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                              : derez.unpack_accessor_RW<T, 3>(regions[0], rect);
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
            ProdReduction<T>::template fold<true /*exclusive*/>(inout[x][y][z], in[x][y][z]);
      break;
    }
    default:
      assert(false);
  }
}

#ifdef LEGATE_USE_OPENMP
template<typename T>
/*static*/ void ProdTask<T>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                         Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          axis          = derez.unpack_dimension();
  const int          collapse_dim  = derez.unpack_dimension();
  const int          init_dim      = derez.unpack_dimension();
  const T            initial_value = (task->futures.size() == 1) ? task->futures[0].get_result<T>() : ProdReduction<T>::identity;
  switch (init_dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<T, 1> out =
          (collapse_dim >= 0) ? derez.unpack_accessor_WO<T, 1>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                              : derez.unpack_accessor_WO<T, 1>(regions[0], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        out[x] = initial_value;
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<T, 2> out =
          (collapse_dim >= 0) ? derez.unpack_accessor_WO<T, 2>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                              : derez.unpack_accessor_WO<T, 2>(regions[0], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          out[x][y] = initial_value;
      break;
    }
    default:
      assert(false);    // shouldn't see any other cases
  }
  const int dim = derez.unpack_dimension();
  switch (dim) {
    // Should never get the case of 1 as this would just be a copy since
    // reducing our only dimension should have called ProdReducTask
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorRW<T, 2> inout =
          (collapse_dim >= 0) ? derez.unpack_accessor_RW<T, 2, 1>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                              : derez.unpack_accessor_RW<T, 2>(regions[0], rect);
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      if (axis == 0) {
// Flip dimension order to avoid races between threads on the collapsing dim
#  pragma omp parallel for
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            ProdReduction<T>::template fold<true /*exclusive*/>(inout[x][y], in[x][y]);
      } else {
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            ProdReduction<T>::template fold<true /*exclusive*/>(inout[x][y], in[x][y]);
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) return;
      const AccessorRW<T, 3> inout =
          (collapse_dim >= 0) ? derez.unpack_accessor_RW<T, 3, 2>(regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                              : derez.unpack_accessor_RW<T, 3>(regions[0], rect);
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      if (axis == 0) {
// Flip dimension order to avoid races between threads on the collapsing dim
#  pragma omp parallel for
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              ProdReduction<T>::template fold<true /*exclusive*/>(inout[x][y][z], in[x][y][z]);
      } else {
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              ProdReduction<T>::template fold<true /*exclusive*/>(inout[x][y][z], in[x][y][z]);
      }
      break;
    }
    default:
      assert(false);
  }
}
#endif

template<typename T>
/*static*/ T ProdReducTask<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                           Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim    = derez.unpack_dimension();
  T                  result = ProdReduction<T>::identity;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        result *= in[x];
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          result *= in[x][y];
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
            result *= in[x][y][z];
      break;
    }
    default:
      assert(false);
  }
  return result;
}

#ifdef LEGATE_USE_OPENMP
template<typename T>
/*static*/ T ProdReducTask<T>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                           Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim         = derez.unpack_dimension();
  const int          max_threads = omp_get_max_threads();
  T*                 results     = (T*)alloca(max_threads * sizeof(T));
  for (int i = 0; i < max_threads; i++)
    results[i] = ProdReduction<T>::identity;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
#  pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#  pragma omp for nowait
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          ProdReduction<T>::template fold<true /*exclusive*/>(results[tid], in[x]);
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[0], rect);
#  pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#  pragma omp for nowait
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            ProdReduction<T>::template fold<true /*exclusive*/>(results[tid], in[x][y]);
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[0], rect);
#  pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#  pragma omp for nowait
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              ProdReduction<T>::template fold<true /*exclusive*/>(results[tid], in[x][y][z]);
      }
      break;
    }
    default:
      assert(false);
  }
  T result = results[0];
  for (int i = 1; i < max_threads; i++)
    ProdReduction<T>::template fold<true /*exclusive*/>(result, results[i]);
  return result;
}
#endif

template<typename T>
/*static*/ void ProdRadixTask<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                              Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  assert(task->regions.size() <= MAX_REDUCTION_RADIX);
  const int     radix         = derez.unpack_dimension();
  const int     extra_dim_out = derez.unpack_dimension();
  const int     extra_dim_in  = derez.unpack_dimension();
  const int     dim           = derez.unpack_dimension();
  const coord_t offset        = (extra_dim_in >= 0) ? task->index_point[extra_dim_in] * radix : 0;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 1> out =
          (extra_dim_out >= 0) ? derez.unpack_accessor_WO<T, 1>(regions[0], rect, extra_dim_out, task->index_point[extra_dim_out])
                               : derez.unpack_accessor_WO<T, 1>(regions[0], rect);
      AccessorRO<T, 1> in[MAX_REDUCTION_RADIX];
      unsigned         num_inputs = 0;
      for (unsigned idx = 1; idx < task->regions.size(); idx++)
        if (task->regions[idx].region.exists())
          in[num_inputs++] = derez.unpack_accessor_RO<T, 1>(regions[idx], rect, extra_dim_in, offset + idx - 1);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
        T val = in[0][x];
        for (unsigned idx = 1; idx < num_inputs; idx++)
          ProdReduction<T>::template fold<true /*exclusive*/>(val, in[idx][x]);
        out[x] = val;
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 2> out =
          (extra_dim_out >= 0) ? derez.unpack_accessor_WO<T, 2>(regions[0], rect, extra_dim_out, task->index_point[extra_dim_out])
                               : derez.unpack_accessor_WO<T, 2>(regions[0], rect);
      AccessorRO<T, 2> in[MAX_REDUCTION_RADIX];
      unsigned         num_inputs = 0;
      for (unsigned idx = 1; idx < task->regions.size(); idx++)
        if (task->regions[idx].region.exists())
          in[num_inputs++] = derez.unpack_accessor_RO<T, 2>(regions[idx], rect, extra_dim_in, offset + idx - 1);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
          T val = in[0][x][y];
          for (unsigned idx = 1; idx < num_inputs; idx++)
            ProdReduction<T>::template fold<true /*exclusive*/>(val, in[idx][x][y]);
          out[x][y] = val;
        }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 3> out =
          (extra_dim_out >= 0) ? derez.unpack_accessor_WO<T, 3>(regions[0], rect, extra_dim_out, task->index_point[extra_dim_out])
                               : derez.unpack_accessor_WO<T, 3>(regions[0], rect);
      AccessorRO<T, 3> in[MAX_REDUCTION_RADIX];
      unsigned         num_inputs = 0;
      for (unsigned idx = 1; idx < task->regions.size(); idx++)
        if (task->regions[idx].region.exists())
          in[num_inputs++] = derez.unpack_accessor_RO<T, 3>(regions[idx], rect, extra_dim_in, offset + idx - 1);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
            T val = in[0][x][y][z];
            for (unsigned idx = 1; idx < num_inputs; idx++)
              ProdReduction<T>::template fold<true /*exclusive*/>(val, in[idx][x][y][z]);
            out[x][y][z] = val;
          }
      break;
    }
    default:
      assert(false);
  }
}

#ifdef LEGATE_USE_OPENMP
template<typename T>
/*static*/ void ProdRadixTask<T>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                              Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  assert(task->regions.size() <= MAX_REDUCTION_RADIX);
  const int     radix         = derez.unpack_dimension();
  const int     extra_dim_out = derez.unpack_dimension();
  const int     extra_dim_in  = derez.unpack_dimension();
  const int     dim           = derez.unpack_dimension();
  const coord_t offset        = (extra_dim_in >= 0) ? task->index_point[extra_dim_in] * radix : 0;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 1> out =
          (extra_dim_out >= 0) ? derez.unpack_accessor_WO<T, 1>(regions[0], rect, extra_dim_out, task->index_point[extra_dim_out])
                               : derez.unpack_accessor_WO<T, 1>(regions[0], rect);
      AccessorRO<T, 1> in[MAX_REDUCTION_RADIX];
      unsigned         num_inputs = 0;
      for (unsigned idx = 1; idx < task->regions.size(); idx++)
        if (task->regions[idx].region.exists())
          in[num_inputs++] = derez.unpack_accessor_RO<T, 1>(regions[idx], rect, extra_dim_in, offset + idx - 1);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
        T val = in[0][x];
        for (unsigned idx = 1; idx < num_inputs; idx++)
          ProdReduction<T>::template fold<true /*exclusive*/>(val, in[idx][x]);
        out[x] = val;
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 2> out =
          (extra_dim_out >= 0) ? derez.unpack_accessor_WO<T, 2>(regions[0], rect, extra_dim_out, task->index_point[extra_dim_out])
                               : derez.unpack_accessor_WO<T, 2>(regions[0], rect);
      AccessorRO<T, 2> in[MAX_REDUCTION_RADIX];
      unsigned         num_inputs = 0;
      for (unsigned idx = 1; idx < task->regions.size(); idx++)
        if (task->regions[idx].region.exists())
          in[num_inputs++] = derez.unpack_accessor_RO<T, 2>(regions[idx], rect, extra_dim_in, offset + idx - 1);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
          T val = in[0][x][y];
          for (unsigned idx = 1; idx < num_inputs; idx++)
            ProdReduction<T>::template fold<true /*exclusive*/>(val, in[idx][x][y]);
          out[x][y] = val;
        }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 3> out =
          (extra_dim_out >= 0) ? derez.unpack_accessor_WO<T, 3>(regions[0], rect, extra_dim_out, task->index_point[extra_dim_out])
                               : derez.unpack_accessor_WO<T, 3>(regions[0], rect);
      AccessorRO<T, 3> in[MAX_REDUCTION_RADIX];
      unsigned         num_inputs = 0;
      for (unsigned idx = 1; idx < task->regions.size(); idx++)
        if (task->regions[idx].region.exists())
          in[num_inputs++] = derez.unpack_accessor_RO<T, 3>(regions[idx], rect, extra_dim_in, offset + idx - 1);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
            T val = in[0][x][y][z];
            for (unsigned idx = 1; idx < num_inputs; idx++)
              ProdReduction<T>::template fold<true /*exclusive*/>(val, in[idx][x][y][z]);
            out[x][y][z] = val;
          }
      break;
    }
    default:
      assert(false);
  }
}
#endif

template<typename T>
/*static*/ T ProdScalarTask<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                            Runtime* runtime) {
  assert(task->futures.size() == 2);
  T       one = task->futures[0].get_result<T>();
  const T two = task->futures[1].get_result<T>();
  ProdReduction<T>::template fold<true /*exclusive*/>(one, two);
  return one;
}

INSTANTIATE_ALL_TASKS(ProdTask, static_cast<int>(NumPyOpCode::NUMPY_PROD) * NUMPY_TYPE_OFFSET)
INSTANTIATE_ALL_TASKS(ProdReducTask, static_cast<int>(NumPyOpCode::NUMPY_PROD) * NUMPY_TYPE_OFFSET + NUMPY_REDUCTION_VARIANT_OFFSET)
INSTANTIATE_ALL_TASKS(ProdRadixTask, static_cast<int>(NumPyOpCode::NUMPY_PROD_RADIX) * NUMPY_TYPE_OFFSET)
INSTANTIATE_ALL_TASKS(ProdScalarTask, static_cast<int>(NumPyOpCode::NUMPY_PROD) * NUMPY_TYPE_OFFSET + NUMPY_SCALAR_VARIANT_OFFSET)

}    // namespace numpy
}    // namespace legate

namespace    // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  REGISTER_ALL_TASKS(legate::numpy::ProdTask)
  REGISTER_ALL_TASKS_WITH_REDUCTION_RETURN(legate::numpy::ProdReducTask, ProdReduction)
  REGISTER_ALL_TASKS(legate::numpy::ProdRadixTask)
  REGISTER_ALL_TASKS_WITH_RETURN(legate::numpy::ProdScalarTask)
}
}    // namespace
