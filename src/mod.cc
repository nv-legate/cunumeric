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

#include "mod.h"
#include "proj.h"
#include <cmath>

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
/*static*/ void IntModTask<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                           Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 2) {
        const AccessorRW<T, 1> out = derez.unpack_accessor_RW<T, 1>(regions[0], rect);
        const AccessorRO<T, 1> in  = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          out[x] %= in[x];
      } else {
        const AccessorWO<T, 1> out = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
        const AccessorRO<T, 1> in1 = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
        const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          out[x] = in1[x] % in2[x];
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 2) {
        const AccessorRW<T, 2> out = derez.unpack_accessor_RW<T, 2>(regions[0], rect);
        const AccessorRO<T, 2> in  = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            out[x][y] %= in[x][y];
      } else {
        const AccessorWO<T, 2> out = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
        const AccessorRO<T, 2> in1 = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
        const AccessorRO<T, 2> in2 = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            out[x][y] = in1[x][y] % in2[x][y];
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 2) {
        const AccessorRW<T, 3> out = derez.unpack_accessor_RW<T, 3>(regions[0], rect);
        const AccessorRO<T, 3> in  = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              out[x][y][z] %= in[x][y][z];
      } else {
        const AccessorWO<T, 3> out = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
        const AccessorRO<T, 3> in1 = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
        const AccessorRO<T, 3> in2 = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              out[x][y][z] = in1[x][y][z] % in2[x][y][z];
      }
      break;
    }
    default:
      assert(false);
  }
}

#ifdef LEGATE_USE_OPENMP
template<typename T>
/*static*/ void IntModTask<T>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                           Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 2) {
        const AccessorRW<T, 1> out = derez.unpack_accessor_RW<T, 1>(regions[0], rect);
        const AccessorRO<T, 1> in  = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          out[x] %= in[x];
      } else {
        const AccessorWO<T, 1> out = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
        const AccessorRO<T, 1> in1 = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
        const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          out[x] = in1[x] % in2[x];
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 2) {
        const AccessorRW<T, 2> out = derez.unpack_accessor_RW<T, 2>(regions[0], rect);
        const AccessorRO<T, 2> in  = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            out[x][y] %= in[x][y];
      } else {
        const AccessorWO<T, 2> out = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
        const AccessorRO<T, 2> in1 = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
        const AccessorRO<T, 2> in2 = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            out[x][y] = in1[x][y] % in2[x][y];
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 2) {
        const AccessorRW<T, 3> out = derez.unpack_accessor_RW<T, 3>(regions[0], rect);
        const AccessorRO<T, 3> in  = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              out[x][y][z] %= in[x][y][z];
      } else {
        const AccessorWO<T, 3> out = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
        const AccessorRO<T, 3> in1 = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
        const AccessorRO<T, 3> in2 = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              out[x][y][z] = in1[x][y][z] % in2[x][y][z];
      }
      break;
    }
    default:
      assert(false);
  }
}
#endif

template<typename T>
T python_mod(const T& a, const T& b) {
  T res = static_cast<T>(std::fmod(a, b));
  if (res) {
    if ((b < T{0}) != (res < T{0})) res += b;
  } else {
    res = std::copysign(T{0}, b);
  }
  return res;
}

template<typename T>
/*static*/ void RealModTask<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                            Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 2) {
        const AccessorRW<T, 1> out = derez.unpack_accessor_RW<T, 1>(regions[0], rect);
        const AccessorRO<T, 1> in  = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          out[x] = python_mod(out[x], in[x]);
      } else {
        const AccessorWO<T, 1> out = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
        const AccessorRO<T, 1> in1 = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
        const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          out[x] = python_mod(in1[x], in2[x]);
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 2) {
        const AccessorRW<T, 2> out = derez.unpack_accessor_RW<T, 2>(regions[0], rect);
        const AccessorRO<T, 2> in  = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            out[x][y] = python_mod(out[x][y], in[x][y]);
      } else {
        const AccessorWO<T, 2> out = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
        const AccessorRO<T, 2> in1 = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
        const AccessorRO<T, 2> in2 = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            out[x][y] = python_mod(in1[x][y], in2[x][y]);
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 2) {
        const AccessorRW<T, 3> out = derez.unpack_accessor_RW<T, 3>(regions[0], rect);
        const AccessorRO<T, 3> in  = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              out[x][y][z] = python_mod(out[x][y][z], in[x][y][z]);
      } else {
        const AccessorWO<T, 3> out = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
        const AccessorRO<T, 3> in1 = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
        const AccessorRO<T, 3> in2 = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              out[x][y][z] = python_mod(in1[x][y][z], in2[x][y][z]);
      }
      break;
    }
    default:
      assert(false);
  }
}

#ifdef LEGATE_USE_OPENMP
template<typename T>
/*static*/ void RealModTask<T>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                            Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 2) {
        const AccessorRW<T, 1> out = derez.unpack_accessor_RW<T, 1>(regions[0], rect);
        const AccessorRO<T, 1> in  = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          out[x] = python_mod(out[x], in[x]);
      } else {
        const AccessorWO<T, 1> out = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
        const AccessorRO<T, 1> in1 = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
        const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          out[x] = python_mod(in1[x], in2[x]);
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 2) {
        const AccessorRW<T, 2> out = derez.unpack_accessor_RW<T, 2>(regions[0], rect);
        const AccessorRO<T, 2> in  = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            out[x][y] = python_mod(out[x][y], in[x][y]);
      } else {
        const AccessorWO<T, 2> out = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
        const AccessorRO<T, 2> in1 = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
        const AccessorRO<T, 2> in2 = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            out[x][y] = python_mod(in1[x][y], in2[x][y]);
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 2) {
        const AccessorRW<T, 3> out = derez.unpack_accessor_RW<T, 3>(regions[0], rect);
        const AccessorRO<T, 3> in  = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              out[x][y][z] = python_mod(out[x][y][z], in[x][y][z]);
      } else {
        const AccessorWO<T, 3> out = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
        const AccessorRO<T, 3> in1 = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
        const AccessorRO<T, 3> in2 = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              out[x][y][z] = python_mod(in1[x][y][z], in2[x][y][z]);
      }
      break;
    }
    default:
      assert(false);
  }
}
#endif

template<typename T>
/*static*/ void IntModBroadcast<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  assert(task->futures.size() == 1);
  const T in2 = task->futures[0].get_result<T>();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 1) {
        const AccessorRW<T, 1> out = derez.unpack_accessor_RW<T, 1>(regions[0], rect);
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          out[x] %= in2;
      } else {
        const AccessorWO<T, 1> out   = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
        const AccessorRO<T, 1> in1   = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
        const unsigned         index = derez.unpack_32bit_uint();
        assert((index == 0) || (index == 1));
        if (index == 0) {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            out[x] = in2 % in1[x];
        } else {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            out[x] = in1[x] % in2;
        }
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 1) {
        const AccessorRW<T, 2> out = derez.unpack_accessor_RW<T, 2>(regions[0], rect);
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            out[x][y] %= in2;
      } else {
        const AccessorWO<T, 2> out   = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
        const AccessorRO<T, 2> in1   = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
        const unsigned         index = derez.unpack_32bit_uint();
        assert((index == 0) || (index == 1));
        if (index == 0) {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              out[x][y] = in2 % in1[x][y];
        } else {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              out[x][y] = in1[x][y] % in2;
        }
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 1) {
        const AccessorRW<T, 3> out = derez.unpack_accessor_RW<T, 3>(regions[0], rect);
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              out[x][y][z] %= in2;
      } else {
        const AccessorWO<T, 3> out   = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
        const AccessorRO<T, 3> in1   = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
        const unsigned         index = derez.unpack_32bit_uint();
        assert((index == 0) || (index == 1));
        if (index == 0) {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                out[x][y][z] = in2 % in1[x][y][z];
        } else {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                out[x][y][z] = in1[x][y][z] % in2;
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
/*static*/ void IntModBroadcast<T>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  assert(task->futures.size() == 1);
  const T in2 = task->futures[0].get_result<T>();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 1) {
        const AccessorRW<T, 1> out = derez.unpack_accessor_RW<T, 1>(regions[0], rect);
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          out[x] %= in2;
      } else {
        const AccessorWO<T, 1> out   = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
        const AccessorRO<T, 1> in1   = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
        const unsigned         index = derez.unpack_32bit_uint();
        assert((index == 0) || (index == 1));
        if (index == 0) {
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            out[x] = in2 % in1[x];
        } else {
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            out[x] = in1[x] % in2;
        }
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 1) {
        const AccessorRW<T, 2> out = derez.unpack_accessor_RW<T, 2>(regions[0], rect);
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            out[x][y] %= in2;
      } else {
        const AccessorWO<T, 2> out   = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
        const AccessorRO<T, 2> in1   = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
        const unsigned         index = derez.unpack_32bit_uint();
        assert((index == 0) || (index == 1));
        if (index == 0) {
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              out[x][y] = in2 % in1[x][y];
        } else {
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              out[x][y] = in1[x][y] % in2;
        }
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 1) {
        const AccessorRW<T, 3> out = derez.unpack_accessor_RW<T, 3>(regions[0], rect);
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              out[x][y][z] %= in2;
      } else {
        const AccessorWO<T, 3> out   = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
        const AccessorRO<T, 3> in1   = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
        const unsigned         index = derez.unpack_32bit_uint();
        assert((index == 0) || (index == 1));
        if (index == 0) {
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                out[x][y][z] = in2 % in1[x][y][z];
        } else {
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                out[x][y][z] = in1[x][y][z] % in2;
        }
      }
      break;
    }
    default:
      assert(false);
  }
}
#endif

template<typename T>
/*static*/ void RealModBroadcast<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                 Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  assert(task->futures.size() == 1);
  const T in2 = task->futures[0].get_result<T>();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 1) {
        const AccessorRW<T, 1> out = derez.unpack_accessor_RW<T, 1>(regions[0], rect);
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          out[x] = python_mod(out[x], in2);
      } else {
        const AccessorWO<T, 1> out   = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
        const AccessorRO<T, 1> in1   = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
        const unsigned         index = derez.unpack_32bit_uint();
        assert((index == 0) || (index == 1));
        if (index == 0) {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            out[x] = python_mod(in2, in1[x]);
        } else {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            out[x] = python_mod(in1[x], in2);
        }
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 1) {
        const AccessorRW<T, 2> out = derez.unpack_accessor_RW<T, 2>(regions[0], rect);
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            out[x][y] = python_mod(out[x][y], in2);
      } else {
        const AccessorWO<T, 2> out   = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
        const AccessorRO<T, 2> in1   = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
        const unsigned         index = derez.unpack_32bit_uint();
        assert((index == 0) || (index == 1));
        if (index == 0) {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              out[x][y] = python_mod(in2, in1[x][y]);
        } else {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              out[x][y] = python_mod(in1[x][y], in2);
        }
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 1) {
        const AccessorRW<T, 3> out = derez.unpack_accessor_RW<T, 3>(regions[0], rect);
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              out[x][y][z] = python_mod(out[x][y][z], in2);
      } else {
        const AccessorWO<T, 3> out   = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
        const AccessorRO<T, 3> in1   = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
        const unsigned         index = derez.unpack_32bit_uint();
        assert((index == 0) || (index == 1));
        if (index == 0) {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                out[x][y][z] = python_mod(in2, in1[x][y][z]);
        } else {
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                out[x][y][z] = python_mod(in1[x][y][z], in2);
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
/*static*/ void RealModBroadcast<T>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                 Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  assert(task->futures.size() == 1);
  const T in2 = task->futures[0].get_result<T>();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 1) {
        const AccessorRW<T, 1> out = derez.unpack_accessor_RW<T, 1>(regions[0], rect);
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          out[x] = python_mod(out[x], in2);
      } else {
        const AccessorWO<T, 1> out   = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
        const AccessorRO<T, 1> in1   = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
        const unsigned         index = derez.unpack_32bit_uint();
        assert((index == 0) || (index == 1));
        if (index == 0) {
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            out[x] = python_mod(in2, in1[x]);
        } else {
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            out[x] = python_mod(in1[x], in2);
        }
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 1) {
        const AccessorRW<T, 2> out = derez.unpack_accessor_RW<T, 2>(regions[0], rect);
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            out[x][y] = python_mod(out[x][y], in2);
      } else {
        const AccessorWO<T, 2> out   = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
        const AccessorRO<T, 2> in1   = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
        const unsigned         index = derez.unpack_32bit_uint();
        assert((index == 0) || (index == 1));
        if (index == 0) {
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              out[x][y] = python_mod(in2, in1[x][y]);
        } else {
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              out[x][y] = python_mod(in1[x][y], in2);
        }
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 1) {
        const AccessorRW<T, 3> out = derez.unpack_accessor_RW<T, 3>(regions[0], rect);
#  pragma omp parallel for
        for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
          for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
            for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
              out[x][y][z] = python_mod(out[x][y][z], in2);
      } else {
        const AccessorWO<T, 3> out   = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
        const AccessorRO<T, 3> in1   = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
        const unsigned         index = derez.unpack_32bit_uint();
        assert((index == 0) || (index == 1));
        if (index == 0) {
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                out[x][y][z] = python_mod(in2, in1[x][y][z]);
        } else {
#  pragma omp parallel for
          for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
            for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
              for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++)
                out[x][y][z] = python_mod(in1[x][y][z], in2);
        }
      }
      break;
    }
    default:
      assert(false);
  }
}
#endif

template<typename T>
/*static*/ T IntModScalar<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                          Runtime* runtime) {
  assert(task->futures.size() == 2);
  T one = task->futures[0].get_result<T>();
  T two = task->futures[1].get_result<T>();
  return one % two;
}

template<typename T>
/*static*/ T RealModScalar<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                           Runtime* runtime) {
  assert(task->futures.size() == 2);
  T one = task->futures[0].get_result<T>();
  T two = task->futures[1].get_result<T>();
  return python_mod(one, two);
}

INSTANTIATE_INT_TASKS(IntModTask, static_cast<int>(NumPyOpCode::NUMPY_MOD) * NUMPY_TYPE_OFFSET + NUMPY_NORMAL_VARIANT_OFFSET)
INSTANTIATE_REAL_TASKS(RealModTask, static_cast<int>(NumPyOpCode::NUMPY_MOD) * NUMPY_TYPE_OFFSET + NUMPY_NORMAL_VARIANT_OFFSET)
INSTANTIATE_INT_TASKS(IntModBroadcast,
                      static_cast<int>(NumPyOpCode::NUMPY_MOD) * NUMPY_TYPE_OFFSET + NUMPY_BROADCAST_VARIANT_OFFSET)
INSTANTIATE_REAL_TASKS(RealModBroadcast,
                       static_cast<int>(NumPyOpCode::NUMPY_MOD) * NUMPY_TYPE_OFFSET + NUMPY_BROADCAST_VARIANT_OFFSET)
INSTANTIATE_INT_TASKS(IntModScalar, static_cast<int>(NumPyOpCode::NUMPY_MOD) * NUMPY_TYPE_OFFSET + NUMPY_SCALAR_VARIANT_OFFSET)
INSTANTIATE_REAL_TASKS(RealModScalar, static_cast<int>(NumPyOpCode::NUMPY_MOD) * NUMPY_TYPE_OFFSET + NUMPY_SCALAR_VARIANT_OFFSET)

}    // namespace numpy
}    // namespace legate

namespace    // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  REGISTER_INT_TASKS(legate::numpy::IntModTask)
  REGISTER_REAL_TASKS(legate::numpy::RealModTask)
  REGISTER_INT_TASKS(legate::numpy::IntModBroadcast)
  REGISTER_REAL_TASKS(legate::numpy::RealModBroadcast)
  REGISTER_INT_TASKS_WITH_RETURN(legate::numpy::IntModScalar)
  REGISTER_REAL_TASKS_WITH_RETURN(legate::numpy::RealModScalar)
}
}    // namespace
