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

#include "argmin.h"
#include "proj.h"
#ifdef LEGATE_USE_OPENMP
#include <omp.h>
#endif

using namespace Legion;

namespace legate {
namespace numpy {

/*static*/ const Argval<__half> ArgminReduction<__half>::identity =
  Argval<__half>(MinReduction<__half>::identity);
/*static*/ const Argval<float> ArgminReduction<float>::identity =
  Argval<float>(MinReduction<float>::identity);
/*static*/ const Argval<double> ArgminReduction<double>::identity =
  Argval<double>(MinReduction<double>::identity);

/*static*/ const Argval<int16_t> ArgminReduction<int16_t>::identity =
  Argval<int16_t>(MinReduction<int16_t>::identity);
/*static*/ const Argval<int32_t> ArgminReduction<int32_t>::identity =
  Argval<int32_t>(MinReduction<int32_t>::identity);
/*static*/ const Argval<int64_t> ArgminReduction<int64_t>::identity =
  Argval<int64_t>(MinReduction<int64_t>::identity);

/*static*/ const Argval<uint16_t> ArgminReduction<uint16_t>::identity =
  Argval<uint16_t>(MinReduction<uint16_t>::identity);
/*static*/ const Argval<uint32_t> ArgminReduction<uint32_t>::identity =
  Argval<uint32_t>(MinReduction<uint32_t>::identity);
/*static*/ const Argval<uint64_t> ArgminReduction<uint64_t>::identity =
  Argval<uint64_t>(MinReduction<uint64_t>::identity);

// /*static*/ const Argval<complex<__half>> ArgminReduction<complex<__half>>::identity =
//     Argval<complex<__half>>(MinReduction<complex<__half>>::identity);
/*static*/ const Argval<complex<float>> ArgminReduction<complex<float>>::identity =
  Argval<complex<float>>(MinReduction<complex<float>>::identity);
// /*static*/ const Argval<complex<double>> ArgminReduction<complex<double>>::identity =
// Argval<complex<double>>(MinReduction<complex<double>>::identity);

/*static*/ const Argval<bool> ArgminReduction<bool>::identity =
  Argval<bool>(MinReduction<bool>::identity);

template <typename T>
/*static*/ void ArgminTask<T>::cpu_variant(const Task* task,
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
      const AccessorWO<Argval<T>, 1> out =
        (collapse_dim >= 0) ? derez.unpack_accessor_WO<Argval<T>, 1>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_WO<Argval<T>, 1>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) out[x] = ArgminReduction<T>::identity;
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<Argval<T>, 2> out =
        (collapse_dim >= 0) ? derez.unpack_accessor_WO<Argval<T>, 2>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_WO<Argval<T>, 2>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) out[x][y] = ArgminReduction<T>::identity;
      break;
    }
    default: assert(false);  // shouldn't see any other cases
  }
  const int dim = derez.unpack_dimension();
  switch (dim) {
    // Should never get the case of 1 as this would just be a copy since
    // reducing our only dimension should have called MinReducTask
    case 2: {
      assert((axis == 0) || (axis == 1));
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorRW<Argval<T>, 2> inout =
        (collapse_dim >= 0) ? derez.unpack_accessor_RW<Argval<T>, 2, 1>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_RW<Argval<T>, 2>(regions[0], rect);
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
          const Argval<T> value(axis == 0 ? x : y, in[x][y]);
          ArgminReduction<T>::template fold<true /*exclusive*/>(inout[x][y], value);
        }
      break;
    }
    case 3: {
      assert((axis == 0) || (axis == 1) || (axis == 2));
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) return;
      const AccessorRW<Argval<T>, 3> inout =
        (collapse_dim >= 0) ? derez.unpack_accessor_RW<Argval<T>, 3, 2>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_RW<Argval<T>, 3>(regions[0], rect);
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
            const Argval<T> value(axis == 0 ? x : (axis == 1) ? y : z, in[x][y][z]);
            ArgminReduction<T>::template fold<true /*exclusive*/>(inout[x][y][z], value);
          }
      break;
    }
    default: assert(false);
  }
}

#ifdef LEGATE_USE_OPENMP
template <typename T>
/*static*/ void ArgminTask<T>::omp_variant(const Task* task,
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
      const AccessorWO<Argval<T>, 1> out =
        (collapse_dim >= 0) ? derez.unpack_accessor_WO<Argval<T>, 1>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_WO<Argval<T>, 1>(regions[0], rect);
#pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) out[x] = ArgminReduction<T>::identity;
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorWO<Argval<T>, 2> out =
        (collapse_dim >= 0) ? derez.unpack_accessor_WO<Argval<T>, 2>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_WO<Argval<T>, 2>(regions[0], rect);
#pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) out[x][y] = ArgminReduction<T>::identity;
      break;
    }
    default: assert(false);  // shouldn't see any other cases
  }
  const int dim = derez.unpack_dimension();
  switch (dim) {
    // Should never get the case of 1 as this would just be a copy since
    // reducing our only dimension should have called MinReducTask
    case 2: {
      assert((axis == 0) || (axis == 1));
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) return;
      const AccessorRW<Argval<T>, 2> inout =
        (collapse_dim >= 0) ? derez.unpack_accessor_RW<Argval<T>, 2, 1>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_RW<Argval<T>, 2>(regions[0], rect);
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      if (collapse_dim == 0) {
// Flip dimension order to avoid races between threads on the collapsing dim
#pragma omp parallel for
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
            const Argval<T> value(axis == 0 ? x : y, in[x][y]);
            ArgminReduction<T>::template fold<false /*exclusive*/>(inout[x][y], value);
          }
      } else {
#pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
            const Argval<T> value(axis == 0 ? x : y, in[x][y]);
            ArgminReduction<T>::template fold<false /*exclusive*/>(inout[x][y], value);
          }
      }
      break;
    }
    case 3: {
      assert((axis == 0) || (axis == 1) || (axis == 2));
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) return;
      const AccessorRW<Argval<T>, 3> inout =
        (collapse_dim >= 0) ? derez.unpack_accessor_RW<Argval<T>, 3, 2>(
                                regions[0], rect, collapse_dim, task->index_point[collapse_dim])
                            : derez.unpack_accessor_RW<Argval<T>, 3>(regions[0], rect);
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      if (collapse_dim == 0) {
// Flip dimension order to avoid races between threads on the collapsing dim
#pragma omp parallel for
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
              const Argval<T> value(axis == 0 ? x : (axis == 1) ? y : z, in[x][y][z]);
              ArgminReduction<T>::template fold<false /*exclusive*/>(inout[x][y][z], value);
            }
      } else {
#pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
              const Argval<T> value(axis == 0 ? x : (axis == 1) ? y : z, in[x][y][z]);
              ArgminReduction<T>::template fold<false /*exclusive*/>(inout[x][y][z], value);
            }
      }
      break;
    }
    default: assert(false);
  }
}
#endif

template <typename T>
/*static*/ Argval<T> ArgminReducTask<T>::cpu_variant(const Task* task,
                                                     const std::vector<PhysicalRegion>& regions,
                                                     Context ctx,
                                                     Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int dim    = derez.unpack_dimension();
  Argval<T> result = ArgminReduction<T>::identity;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
        const Argval<T> value(x, in[x]);
        ArgminReduction<T>::template fold<true /*exclusive*/>(result, value);
      }
      break;
    }
    default: assert(false);  // shouldn't get any other dimensions here
  }
  return result;
}

#ifdef LEGATE_USE_OPENMP
template <typename T>
/*static*/ Argval<T> ArgminReducTask<T>::omp_variant(const Task* task,
                                                     const std::vector<PhysicalRegion>& regions,
                                                     Context ctx,
                                                     Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int dim    = derez.unpack_dimension();
  Argval<T> result = ArgminReduction<T>::identity;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
#pragma omp parallel
      {
        Argval<T> local = ArgminReduction<T>::identity;
#pragma omp for nowait
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
          const Argval<T> value(x, in[x]);
          ArgminReduction<T>::template fold<true /*exclusive*/>(local, value);
        }
        ArgminReduction<T>::template fold<false /*exclusive*/>(result, local);
      }
      break;
    }
    default: assert(false);  // shouldn't get any other dimensions here
  }
  return result;
}
#endif

template <typename T>
/*static*/ void ArgminRadixTask<T>::cpu_variant(const Task* task,
                                                const std::vector<PhysicalRegion>& regions,
                                                Context ctx,
                                                Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  assert(task->regions.size() <= MAX_REDUCTION_RADIX);
  const int radix         = derez.unpack_dimension();
  const int extra_dim_out = derez.unpack_dimension();
  const int extra_dim_in  = derez.unpack_dimension();
  const int dim           = derez.unpack_dimension();
  const coord_t offset    = (extra_dim_in >= 0) ? task->index_point[extra_dim_in] * radix : 0;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<Argval<T>, 1> out =
        (extra_dim_out >= 0) ? derez.unpack_accessor_WO<Argval<T>, 1>(
                                 regions[0], rect, extra_dim_out, task->index_point[extra_dim_out])
                             : derez.unpack_accessor_WO<Argval<T>, 1>(regions[0], rect);
      AccessorRO<Argval<T>, 1> in[MAX_REDUCTION_RADIX];
      unsigned num_inputs = 0;
      for (unsigned idx = 1; idx < task->regions.size(); idx++)
        if (task->regions[idx].region.exists())
          in[num_inputs++] = derez.unpack_accessor_RO<Argval<T>, 1>(
            regions[idx], rect, extra_dim_in, offset + idx - 1);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
        Argval<T> val = in[0][x];
        for (unsigned idx = 1; idx < num_inputs; idx++)
          ArgminReduction<T>::template fold<true /*exclusive*/>(val, in[idx][x]);
        out[x] = val;
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<Argval<T>, 2> out =
        (extra_dim_out >= 0) ? derez.unpack_accessor_WO<Argval<T>, 2>(
                                 regions[0], rect, extra_dim_out, task->index_point[extra_dim_out])
                             : derez.unpack_accessor_WO<Argval<T>, 2>(regions[0], rect);
      AccessorRO<Argval<T>, 2> in[MAX_REDUCTION_RADIX];
      unsigned num_inputs = 0;
      for (unsigned idx = 1; idx < task->regions.size(); idx++)
        if (task->regions[idx].region.exists())
          in[num_inputs++] = derez.unpack_accessor_RO<Argval<T>, 2>(
            regions[idx], rect, extra_dim_in, offset + idx - 1);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
          Argval<T> val = in[0][x][y];
          for (unsigned idx = 1; idx < num_inputs; idx++)
            ArgminReduction<T>::template fold<true /*exclusive*/>(val, in[idx][x][y]);
          out[x][y] = val;
        }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<Argval<T>, 3> out =
        (extra_dim_out >= 0) ? derez.unpack_accessor_WO<Argval<T>, 3>(
                                 regions[0], rect, extra_dim_out, task->index_point[extra_dim_out])
                             : derez.unpack_accessor_WO<Argval<T>, 3>(regions[0], rect);
      AccessorRO<Argval<T>, 3> in[MAX_REDUCTION_RADIX];
      unsigned num_inputs = 0;
      for (unsigned idx = 1; idx < task->regions.size(); idx++)
        if (task->regions[idx].region.exists())
          in[num_inputs++] = derez.unpack_accessor_RO<Argval<T>, 3>(
            regions[idx], rect, extra_dim_in, offset + idx - 1);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
            Argval<T> val = in[0][x][y][z];
            for (unsigned idx = 1; idx < num_inputs; idx++)
              ArgminReduction<T>::template fold<true /*exclusive*/>(val, in[idx][x][y][z]);
            out[x][y][z] = val;
          }
      break;
    }
    default: assert(false);
  }
}

#ifdef LEGATE_USE_OPENMP
template <typename T>
/*static*/ void ArgminRadixTask<T>::omp_variant(const Task* task,
                                                const std::vector<PhysicalRegion>& regions,
                                                Context ctx,
                                                Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  assert(task->regions.size() < MAX_REDUCTION_RADIX);
  const int radix         = derez.unpack_dimension();
  const int extra_dim_out = derez.unpack_dimension();
  const int extra_dim_in  = derez.unpack_dimension();
  const int dim           = derez.unpack_dimension();
  const coord_t offset    = (extra_dim_in >= 0) ? task->index_point[extra_dim_in] * radix : 0;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<Argval<T>, 1> out =
        (extra_dim_out >= 0) ? derez.unpack_accessor_WO<Argval<T>, 1>(
                                 regions[0], rect, extra_dim_out, task->index_point[extra_dim_out])
                             : derez.unpack_accessor_WO<Argval<T>, 1>(regions[0], rect);
      AccessorRO<Argval<T>, 1> in[MAX_REDUCTION_RADIX];
      unsigned num_inputs = 0;
      for (unsigned idx = 1; idx < task->regions.size(); idx++)
        if (task->regions[idx].region.exists())
          in[num_inputs++] = derez.unpack_accessor_RO<Argval<T>, 1>(
            regions[idx], rect, extra_dim_in, offset + idx - 1);
#pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
        Argval<T> val = in[0][x];
        for (unsigned idx = 1; idx < num_inputs; idx++)
          ArgminReduction<T>::template fold<true /*exclusive*/>(val, in[idx][x]);
        out[x] = val;
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<Argval<T>, 2> out =
        (extra_dim_out >= 0) ? derez.unpack_accessor_WO<Argval<T>, 2>(
                                 regions[0], rect, extra_dim_out, task->index_point[extra_dim_out])
                             : derez.unpack_accessor_WO<Argval<T>, 2>(regions[0], rect);
      AccessorRO<Argval<T>, 2> in[MAX_REDUCTION_RADIX];
      unsigned num_inputs = 0;
      for (unsigned idx = 1; idx < task->regions.size(); idx++)
        if (task->regions[idx].region.exists())
          in[num_inputs++] = derez.unpack_accessor_RO<Argval<T>, 2>(
            regions[idx], rect, extra_dim_in, offset + idx - 1);
#pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
          Argval<T> val = in[0][x][y];
          for (unsigned idx = 1; idx < num_inputs; idx++)
            ArgminReduction<T>::template fold<true /*exclusive*/>(val, in[idx][x][y]);
          out[x][y] = val;
        }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<Argval<T>, 3> out =
        (extra_dim_out >= 0) ? derez.unpack_accessor_WO<Argval<T>, 3>(
                                 regions[0], rect, extra_dim_out, task->index_point[extra_dim_out])
                             : derez.unpack_accessor_WO<Argval<T>, 3>(regions[0], rect);
      AccessorRO<Argval<T>, 3> in[MAX_REDUCTION_RADIX];
      unsigned num_inputs = 0;
      for (unsigned idx = 1; idx < task->regions.size(); idx++)
        if (task->regions[idx].region.exists())
          in[num_inputs++] = derez.unpack_accessor_RO<Argval<T>, 3>(
            regions[idx], rect, extra_dim_in, offset + idx - 1);
#pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
            Argval<T> val = in[0][x][y][z];
            for (unsigned idx = 1; idx < num_inputs; idx++)
              ArgminReduction<T>::template fold<true /*exclusive*/>(val, in[idx][x][y][z]);
            out[x][y][z] = val;
          }
      break;
    }
    default: assert(false);
  }
}
#endif

INSTANTIATE_ALL_TASKS(ArgminTask, static_cast<int>(NumPyOpCode::NUMPY_ARGMIN) * NUMPY_TYPE_OFFSET)
INSTANTIATE_ALL_TASKS(ArgminReducTask,
                      static_cast<int>(NumPyOpCode::NUMPY_ARGMIN) * NUMPY_TYPE_OFFSET +
                        NUMPY_REDUCTION_VARIANT_OFFSET)
INSTANTIATE_ALL_TASKS(ArgminRadixTask,
                      static_cast<int>(NumPyOpCode::NUMPY_ARGMIN_RADIX) * NUMPY_TYPE_OFFSET)

}  // namespace numpy
}  // namespace legate

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  REGISTER_ALL_TASKS(legate::numpy::ArgminTask)
  REGISTER_ALL_TASKS_WITH_WRAP_REDUCTION_RETURN(
    legate::numpy::ArgminReducTask, legate::numpy::Argval, legate::numpy::ArgminReduction)
  REGISTER_ALL_TASKS(legate::numpy::ArgminRadixTask)
}
}  // namespace
