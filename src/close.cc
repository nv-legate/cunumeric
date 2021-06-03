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

#include "close.h"
#include "proj.h"
#include <cmath>

using namespace Legion;

namespace legate {
namespace numpy {

template <typename T>
/*static*/ bool CloseTask<T>::cpu_variant(const Task* task,
                                          const std::vector<PhysicalRegion>& regions,
                                          Context ctx,
                                          Runtime* runtime)
{
  assert(task->futures.size() == 2);
  const double rtol = task->futures[0].get_result<double>();
  const double atol = task->futures[1].get_result<double>();
  LegateDeserializer derez(task->args, task->arglen);
  const int dim = derez.unpack_dimension();
  bool result   = true;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in1 = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
      const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        if (fabs(in1[x] - in2[x]) > (atol + rtol * fabs(in2[x]))) {
          result = false;
          break;
        }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2> in1 = derez.unpack_accessor_RO<T, 2>(regions[0], rect);
      const AccessorRO<T, 2> in2 = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          if (fabs(in1[x][y] - in2[x][y]) > (atol + rtol * fabs(in2[x][y]))) {
            result = false;
            break;
          }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3> in1 = derez.unpack_accessor_RO<T, 3>(regions[0], rect);
      const AccessorRO<T, 3> in2 = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
            if (fabs(in1[x][y][z] - in2[x][y][z]) > (atol + rtol * fabs(in2[x][y][z]))) {
              result = false;
              break;
            }
      break;
    }
    default: assert(false);
  }
  return result;
}

#ifdef LEGATE_USE_OPENMP
template <typename T>
/*static*/ bool CloseTask<T>::omp_variant(const Task* task,
                                          const std::vector<PhysicalRegion>& regions,
                                          Context ctx,
                                          Runtime* runtime)
{
  assert(task->futures.size() == 2);
  const double rtol = task->futures[0].get_result<double>();
  const double atol = task->futures[1].get_result<double>();
  LegateDeserializer derez(task->args, task->arglen);
  const int dim = derez.unpack_dimension();
  bool result   = true;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in1 = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
      const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
#pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        if (fabs(in1[x] - in2[x]) > (atol + rtol * fabs(in2[x]))) result = false;
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2> in1 = derez.unpack_accessor_RO<T, 2>(regions[0], rect);
      const AccessorRO<T, 2> in2 = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
#pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          if (fabs(in1[x][y] - in2[x][y]) > (atol + rtol * fabs(in2[x][y]))) result = false;
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3> in1 = derez.unpack_accessor_RO<T, 3>(regions[0], rect);
      const AccessorRO<T, 3> in2 = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
#pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
            if (fabs(in1[x][y][z] - in2[x][y][z]) > (atol + rtol * fabs(in2[x][y][z])))
              result = false;
      break;
    }
    default: assert(false);
  }
  return result;
}
#endif

template <typename T>
/*static*/ bool CloseBroadcast<T>::cpu_variant(const Task* task,
                                               const std::vector<PhysicalRegion>& regions,
                                               Context ctx,
                                               Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int dim = derez.unpack_dimension();
  assert(task->futures.size() == 3);
  const T in2       = task->futures[0].get_result<T>();
  const double rtol = task->futures[1].get_result<double>();
  const double atol = task->futures[2].get_result<double>();
  bool result       = true;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in1 = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
      const unsigned index       = derez.unpack_32bit_uint();
      assert((index == 0) || (index == 1));
      if (index == 0) {
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          if (fabs(in2 - in1[x]) > (atol + rtol * fabs(in1[x]))) {
            result = false;
            break;
          }
      } else {
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          if (fabs(in1[x] - in2) > (atol + rtol * fabs(in2))) {
            result = false;
            break;
          }
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2> in1 = derez.unpack_accessor_RO<T, 2>(regions[0], rect);
      const unsigned index       = derez.unpack_32bit_uint();
      assert((index == 0) || (index == 1));
      if (index == 0) {
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            if (fabs(in2 - in1[x][y]) > (atol + rtol * fabs(in1[x][y]))) {
              result = false;
              break;
            }
      } else {
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            if (fabs(in1[x][y] - in2) > (atol + rtol * fabs(in2))) {
              result = false;
              break;
            }
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3> in1 = derez.unpack_accessor_RO<T, 3>(regions[0], rect);
      const unsigned index       = derez.unpack_32bit_uint();
      assert((index == 0) || (index == 1));
      if (index == 0) {
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              if (fabs(in2 - in1[x][y][z]) > (atol + rtol * fabs(in1[x][y][z]))) {
                result = false;
                break;
              }
      } else {
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              if (fabs(in1[x][y][z] - in2) > (atol + rtol * fabs(in2))) {
                result = false;
                break;
              }
      }
      break;
    }
    default: assert(false);
  }
  return result;
}

#ifdef LEGATE_USE_OPENMP
template <typename T>
/*static*/ bool CloseBroadcast<T>::omp_variant(const Task* task,
                                               const std::vector<PhysicalRegion>& regions,
                                               Context ctx,
                                               Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int dim = derez.unpack_dimension();
  assert(task->futures.size() == 3);
  const T in2       = task->futures[0].get_result<T>();
  const double rtol = task->futures[1].get_result<double>();
  const double atol = task->futures[2].get_result<double>();
  bool result       = true;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 1> in1 = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
      const unsigned index       = derez.unpack_32bit_uint();
      assert((index == 0) || (index == 1));
      if (index == 0) {
#pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          if (fabs(in2 - in1[x]) > (atol + rtol * fabs(in1[x]))) result = false;
      } else {
#pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          if (fabs(in1[x] - in2) > (atol + rtol * fabs(in2))) result = false;
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 2> in1 = derez.unpack_accessor_RO<T, 2>(regions[0], rect);
      const unsigned index       = derez.unpack_32bit_uint();
      assert((index == 0) || (index == 1));
      if (index == 0) {
#pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            if (fabs(in2 - in1[x][y]) > (atol + rtol * fabs(in1[x][y]))) result = false;
      } else {
#pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            if (fabs(in1[x][y] - in2) > (atol + rtol * fabs(in2))) result = false;
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorRO<T, 3> in1 = derez.unpack_accessor_RO<T, 3>(regions[0], rect);
      const unsigned index       = derez.unpack_32bit_uint();
      assert((index == 0) || (index == 1));
      if (index == 0) {
#pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              if (fabs(in2 - in1[x][y][z]) > (atol + rtol * fabs(in1[x][y][z]))) result = false;
      } else {
#pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              if (fabs(in1[x][y][z] - in2) > (atol + rtol * fabs(in2))) result = false;
      }
      break;
    }
    default: assert(false);
  }
  return result;
}
#endif

template <typename T>
/*static*/ bool CloseScalar<T>::cpu_variant(const Task* task,
                                            const std::vector<PhysicalRegion>& regions,
                                            Context ctx,
                                            Runtime* runtime)
{
  assert(task->futures.size() == 4);
  T one             = task->futures[0].get_result<T>();
  T two             = task->futures[1].get_result<T>();
  const double rtol = task->futures[1].get_result<double>();
  const double atol = task->futures[2].get_result<double>();
  return (fabs(one - two) <= (atol + rtol * fabs(two)));
}

INSTANTIATE_ALL_TASKS(CloseTask,
                      static_cast<int>(NumPyOpCode::NUMPY_ALLCLOSE) * NUMPY_TYPE_OFFSET +
                        NUMPY_REDUCTION_VARIANT_OFFSET)
INSTANTIATE_ALL_TASKS(CloseBroadcast,
                      static_cast<int>(NumPyOpCode::NUMPY_ALLCLOSE) * NUMPY_TYPE_OFFSET +
                        NUMPY_BROADCAST_VARIANT_OFFSET)
INSTANTIATE_ALL_TASKS(CloseScalar,
                      static_cast<int>(NumPyOpCode::NUMPY_ALLCLOSE) * NUMPY_TYPE_OFFSET +
                        NUMPY_SCALAR_VARIANT_OFFSET)

}  // namespace numpy
}  // namespace legate

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  REGISTER_ALL_TASKS_WITH_BOOL_RETURN(legate::numpy::CloseTask)
  REGISTER_ALL_TASKS_WITH_BOOL_RETURN(legate::numpy::CloseBroadcast)
  REGISTER_ALL_TASKS_WITH_BOOL_RETURN(legate::numpy::CloseScalar)
}
}  // namespace
