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

#include "where.h"
#include "proj.h"

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
/*static*/ void WhereTask<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                          Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 1>    out  = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
      const AccessorRO<bool, 1> cond = derez.unpack_accessor_RO<bool, 1>(regions[1], rect);
      const AccessorRO<T, 1>    in1  = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
      const AccessorRO<T, 1>    in2  = derez.unpack_accessor_RO<T, 1>(regions[3], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        out[x] = cond[x] ? in1[x] : in2[x];
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 2>    out  = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
      const AccessorRO<bool, 2> cond = derez.unpack_accessor_RO<bool, 2>(regions[1], rect);
      const AccessorRO<T, 2>    in1  = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
      const AccessorRO<T, 2>    in2  = derez.unpack_accessor_RO<T, 2>(regions[3], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          out[x][y] = cond[x][y] ? in1[x][y] : in2[x][y];
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 3>    out  = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
      const AccessorRO<bool, 3> cond = derez.unpack_accessor_RO<bool, 3>(regions[1], rect);
      const AccessorRO<T, 3>    in1  = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
      const AccessorRO<T, 3>    in2  = derez.unpack_accessor_RO<T, 3>(regions[3], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
            out[x][y][z] = cond[x][y][z] ? in1[x][y][z] : in2[x][y][z];
      break;
    }
    default:
      assert(false);
  }
}

#ifdef LEGATE_USE_OPENMP
template<typename T>
/*static*/ void WhereTask<T>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                          Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 1>    out  = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
      const AccessorRO<bool, 1> cond = derez.unpack_accessor_RO<bool, 1>(regions[1], rect);
      const AccessorRO<T, 1>    in1  = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
      const AccessorRO<T, 1>    in2  = derez.unpack_accessor_RO<T, 1>(regions[3], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        out[x] = cond[x] ? in1[x] : in2[x];
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 2>    out  = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
      const AccessorRO<bool, 2> cond = derez.unpack_accessor_RO<bool, 2>(regions[1], rect);
      const AccessorRO<T, 2>    in1  = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
      const AccessorRO<T, 2>    in2  = derez.unpack_accessor_RO<T, 2>(regions[3], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          out[x][y] = cond[x][y] ? in1[x][y] : in2[x][y];
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 3>    out  = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
      const AccessorRO<bool, 3> cond = derez.unpack_accessor_RO<bool, 3>(regions[1], rect);
      const AccessorRO<T, 3>    in1  = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
      const AccessorRO<T, 3>    in2  = derez.unpack_accessor_RO<T, 3>(regions[3], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
            out[x][y][z] = cond[x][y][z] ? in1[x][y][z] : in2[x][y][z];
      break;
    }
    default:
      assert(false);
  }
}
#endif    // LEGATE_USE_OPNEMP

template<typename T>
/*static*/ void WhereBroadcast<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                               Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim          = derez.unpack_dimension();
  unsigned           future_index = 0;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 1> out         = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
      const bool             future_cond = derez.unpack_bool();
      if (future_cond) {
        const bool condition = task->futures[future_index++].get_result<bool>();
        const bool future1   = derez.unpack_bool();
        if (future1) {
          const T    in1     = task->futures[future_index++].get_result<T>();
          const bool future2 = derez.unpack_bool();
          assert(!future2);    // this would have been the scalar case
          const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
          if (condition) {
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              out[x] = in1;
          } else {
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              out[x] = in2[x];
          }
        } else {
          const AccessorRO<T, 1> in1     = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
          const bool             future2 = derez.unpack_bool();
          if (future2) {
            const T in2 = task->futures[future_index++].get_result<T>();
            if (condition) {
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                out[x] = in1[x];
            } else {
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                out[x] = in2;
            }
          } else {
            const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
            if (condition) {
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                out[x] = in1[x];
            } else {
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                out[x] = in2[x];
            }
          }
        }
      } else {
        const AccessorRO<bool, 1> cond    = derez.unpack_accessor_RO<bool, 1>(regions[1], rect);
        const bool                future1 = derez.unpack_bool();
        if (future1) {
          const T    in1     = task->futures[future_index++].get_result<T>();
          const bool future2 = derez.unpack_bool();
          if (future2) {
            const T in2 = task->futures[future_index++].get_result<T>();
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              out[x] = cond[x] ? in1 : in2;
          } else {
            const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              out[x] = cond[x] ? in1 : in2[x];
          }
        } else {
          const AccessorRO<T, 1> in1     = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
          const bool             future2 = derez.unpack_bool();
          assert(future2);    // this would have been the general case
          const T in2 = task->futures[future_index++].get_result<T>();
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            out[x] = cond[x] ? in1[x] : in2;
        }
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 2> out         = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
      const bool             future_cond = derez.unpack_bool();
      if (future_cond) {
        const bool condition = task->futures[future_index++].get_result<bool>();
        const bool future1   = derez.unpack_bool();
        if (future1) {
          const T    in1     = task->futures[future_index++].get_result<T>();
          const bool future2 = derez.unpack_bool();
          assert(!future2);    // this would have been the scalar case
          const AccessorRO<T, 2> in2 = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
          if (condition) {
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                out[x][y] = in1;
          } else {
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                out[x][y] = in2[x][y];
          }
        } else {
          const AccessorRO<T, 2> in1     = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
          const bool             future2 = derez.unpack_bool();
          if (future2) {
            const T in2 = task->futures[future_index++].get_result<T>();
            if (condition) {
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                  out[x][y] = in1[x][y];
            } else {
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                  out[x][y] = in2;
            }
          } else {
            const AccessorRO<T, 2> in2 = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
            if (condition) {
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                  out[x][y] = in1[x][y];
            } else {
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                  out[x][y] = in2[x][y];
            }
          }
        }
      } else {
        const AccessorRO<bool, 2> cond    = derez.unpack_accessor_RO<bool, 2>(regions[1], rect);
        const bool                future1 = derez.unpack_bool();
        if (future1) {
          const T    in1     = task->futures[future_index++].get_result<T>();
          const bool future2 = derez.unpack_bool();
          if (future2) {
            const T in2 = task->futures[future_index++].get_result<T>();
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                out[x][y] = cond[x][y] ? in1 : in2;
          } else {
            const AccessorRO<T, 2> in2 = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                out[x][y] = cond[x][y] ? in1 : in2[x][y];
          }
        } else {
          const AccessorRO<T, 2> in1     = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
          const bool             future2 = derez.unpack_bool();
          assert(future2);    // this would have been the general case
          const T in2 = task->futures[future_index++].get_result<T>();
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              out[x][y] = cond[x][y] ? in1[x][y] : in2;
        }
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 3> out         = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
      const bool             future_cond = derez.unpack_bool();
      if (future_cond) {
        const bool condition = task->futures[future_index++].get_result<bool>();
        const bool future1   = derez.unpack_bool();
        if (future1) {
          const T    in1     = task->futures[future_index++].get_result<T>();
          const bool future2 = derez.unpack_bool();
          assert(!future2);    // this would have been the scalar case
          const AccessorRO<T, 3> in2 = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
          if (condition) {
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                  out[x][y][z] = in1;
          } else {
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                  out[x][y][z] = in2[x][y][z];
          }
        } else {
          const AccessorRO<T, 3> in1     = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
          const bool             future2 = derez.unpack_bool();
          if (future2) {
            const T in2 = task->futures[future_index++].get_result<T>();
            if (condition) {
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                  for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                    out[x][y][z] = in1[x][y][z];
            } else {
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                  for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                    out[x][y][z] = in2;
            }
          } else {
            const AccessorRO<T, 3> in2 = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
            if (condition) {
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                  for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                    out[x][y][z] = in1[x][y][z];
            } else {
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                  for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                    out[x][y][z] = in2[x][y][z];
            }
          }
        }
      } else {
        const AccessorRO<bool, 3> cond    = derez.unpack_accessor_RO<bool, 3>(regions[1], rect);
        const bool                future1 = derez.unpack_bool();
        if (future1) {
          const T    in1     = task->futures[future_index++].get_result<T>();
          const bool future2 = derez.unpack_bool();
          if (future2) {
            const T in2 = task->futures[future_index++].get_result<T>();
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                  out[x][y][z] = cond[x][y][z] ? in1 : in2;
          } else {
            const AccessorRO<T, 3> in2 = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                  out[x][y][z] = cond[x][y][z] ? in1 : in2[x][y][z];
          }
        } else {
          const AccessorRO<T, 3> in1     = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
          const bool             future2 = derez.unpack_bool();
          assert(future2);    // this would have been the general case
          const T in2 = task->futures[future_index++].get_result<T>();
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                out[x][y][z] = cond[x][y][z] ? in1[x][y][z] : in2;
        }
      }
      break;
    }
    default:
      assert(false);
  }
}

#ifdef LEGATE_USE_OPENMP
template<typename T>
/*static*/ void WhereBroadcast<T>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                               Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim          = derez.unpack_dimension();
  unsigned           future_index = 0;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 1> out         = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
      const bool             future_cond = derez.unpack_bool();
      if (future_cond) {
        const bool condition = task->futures[future_index++].get_result<bool>();
        const bool future1   = derez.unpack_bool();
        if (future1) {
          const T    in1     = task->futures[future_index++].get_result<T>();
          const bool future2 = derez.unpack_bool();
          assert(!future2);    // this would have been the scalar case
          const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
          if (condition) {
#  pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              out[x] = in1;
          } else {
#  pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              out[x] = in2[x];
          }
        } else {
          const AccessorRO<T, 1> in1     = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
          const bool             future2 = derez.unpack_bool();
          if (future2) {
            const T in2 = task->futures[future_index++].get_result<T>();
            if (condition) {
#  pragma omp parallel for
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                out[x] = in1[x];
            } else {
#  pragma omp parallel for
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                out[x] = in2;
            }
          } else {
            const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
            if (condition) {
#  pragma omp parallel for
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                out[x] = in1[x];
            } else {
#  pragma omp parallel for
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                out[x] = in2[x];
            }
          }
        }
      } else {
        const AccessorRO<bool, 1> cond    = derez.unpack_accessor_RO<bool, 1>(regions[1], rect);
        const bool                future1 = derez.unpack_bool();
        if (future1) {
          const T    in1     = task->futures[future_index++].get_result<T>();
          const bool future2 = derez.unpack_bool();
          if (future2) {
            const T in2 = task->futures[future_index++].get_result<T>();
#  pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              out[x] = cond[x] ? in1 : in2;
          } else {
            const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
#  pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              out[x] = cond[x] ? in1 : in2[x];
          }
        } else {
          const AccessorRO<T, 1> in1     = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
          const bool             future2 = derez.unpack_bool();
          assert(future2);    // this would have been the general case
          const T in2 = task->futures[future_index++].get_result<T>();
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            out[x] = cond[x] ? in1[x] : in2;
        }
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 2> out         = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
      const bool             future_cond = derez.unpack_bool();
      if (future_cond) {
        const bool condition = task->futures[future_index++].get_result<bool>();
        const bool future1   = derez.unpack_bool();
        if (future1) {
          const T    in1     = task->futures[future_index++].get_result<T>();
          const bool future2 = derez.unpack_bool();
          assert(!future2);    // this would have been the scalar case
          const AccessorRO<T, 2> in2 = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
          if (condition) {
#  pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                out[x][y] = in1;
          } else {
#  pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                out[x][y] = in2[x][y];
          }
        } else {
          const AccessorRO<T, 2> in1     = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
          const bool             future2 = derez.unpack_bool();
          if (future2) {
            const T in2 = task->futures[future_index++].get_result<T>();
            if (condition) {
#  pragma omp parallel for
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                  out[x][y] = in1[x][y];
            } else {
#  pragma omp parallel for
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                  out[x][y] = in2;
            }
          } else {
            const AccessorRO<T, 2> in2 = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
            if (condition) {
#  pragma omp parallel for
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                  out[x][y] = in1[x][y];
            } else {
#  pragma omp parallel for
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                  out[x][y] = in2[x][y];
            }
          }
        }
      } else {
        const AccessorRO<bool, 2> cond    = derez.unpack_accessor_RO<bool, 2>(regions[1], rect);
        const bool                future1 = derez.unpack_bool();
        if (future1) {
          const T    in1     = task->futures[future_index++].get_result<T>();
          const bool future2 = derez.unpack_bool();
          if (future2) {
            const T in2 = task->futures[future_index++].get_result<T>();
#  pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                out[x][y] = cond[x][y] ? in1 : in2;
          } else {
            const AccessorRO<T, 2> in2 = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
#  pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                out[x][y] = cond[x][y] ? in1 : in2[x][y];
          }
        } else {
          const AccessorRO<T, 2> in1     = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
          const bool             future2 = derez.unpack_bool();
          assert(future2);    // this would have been the general case
          const T in2 = task->futures[future_index++].get_result<T>();
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              out[x][y] = cond[x][y] ? in1[x][y] : in2;
        }
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 3> out         = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
      const bool             future_cond = derez.unpack_bool();
      if (future_cond) {
        const bool condition = task->futures[future_index++].get_result<bool>();
        const bool future1   = derez.unpack_bool();
        if (future1) {
          const T    in1     = task->futures[future_index++].get_result<T>();
          const bool future2 = derez.unpack_bool();
          assert(!future2);    // this would have been the scalar case
          const AccessorRO<T, 3> in2 = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
          if (condition) {
#  pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                  out[x][y][z] = in1;
          } else {
#  pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                  out[x][y][z] = in2[x][y][z];
          }
        } else {
          const AccessorRO<T, 3> in1     = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
          const bool             future2 = derez.unpack_bool();
          if (future2) {
            const T in2 = task->futures[future_index++].get_result<T>();
            if (condition) {
#  pragma omp parallel for
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                  for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                    out[x][y][z] = in1[x][y][z];
            } else {
#  pragma omp parallel for
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                  for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                    out[x][y][z] = in2;
            }
          } else {
            const AccessorRO<T, 3> in2 = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
            if (condition) {
#  pragma omp parallel for
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                  for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                    out[x][y][z] = in1[x][y][z];
            } else {
#  pragma omp parallel for
              for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
                for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                  for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                    out[x][y][z] = in2[x][y][z];
            }
          }
        }
      } else {
        const AccessorRO<bool, 3> cond    = derez.unpack_accessor_RO<bool, 3>(regions[1], rect);
        const bool                future1 = derez.unpack_bool();
        if (future1) {
          const T    in1     = task->futures[future_index++].get_result<T>();
          const bool future2 = derez.unpack_bool();
          if (future2) {
            const T in2 = task->futures[future_index++].get_result<T>();
#  pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                  out[x][y][z] = cond[x][y][z] ? in1 : in2;
          } else {
            const AccessorRO<T, 3> in2 = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
#  pragma omp parallel for
            for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
              for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
                for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                  out[x][y][z] = cond[x][y][z] ? in1 : in2[x][y][z];
          }
        } else {
          const AccessorRO<T, 3> in1     = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
          const bool             future2 = derez.unpack_bool();
          assert(future2);    // this would have been the general case
          const T in2 = task->futures[future_index++].get_result<T>();
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                out[x][y][z] = cond[x][y][z] ? in1[x][y][z] : in2;
        }
      }
      break;
    }
    default:
      assert(false);
  }
}
#endif    // LEGATE_USE_OPENMP

template<typename T>
/*static*/ T WhereScalar<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                         Runtime* runtime) {
  assert(task->futures.size() == 3);
  const bool cond = task->futures[0].get_result<bool>();
  const T    one  = task->futures[1].get_result<T>();
  const T    two  = task->futures[2].get_result<T>();
  return cond ? one : two;
}

INSTANTIATE_ALL_TASKS(WhereTask, static_cast<int>(NumPyOpCode::NUMPY_WHERE) * NUMPY_TYPE_OFFSET + NUMPY_NORMAL_VARIANT_OFFSET)
INSTANTIATE_ALL_TASKS(WhereBroadcast,
                      static_cast<int>(NumPyOpCode::NUMPY_WHERE) * NUMPY_TYPE_OFFSET + NUMPY_BROADCAST_VARIANT_OFFSET)
INSTANTIATE_ALL_TASKS(WhereScalar, static_cast<int>(NumPyOpCode::NUMPY_WHERE) * NUMPY_TYPE_OFFSET + NUMPY_SCALAR_VARIANT_OFFSET)

}    // namespace numpy
}    // namespace legate

namespace    // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  REGISTER_ALL_TASKS(legate::numpy::WhereTask)
  REGISTER_ALL_TASKS(legate::numpy::WhereBroadcast)
  REGISTER_ALL_TASKS_WITH_RETURN(legate::numpy::WhereScalar)
}
}    // namespace
