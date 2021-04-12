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

#include "rand.h"
#include "proj.h"
#include <math.h>

using namespace Legion;

namespace legate {
namespace numpy {

static inline double erfinv(double a) {
  double                 p, q, t, fa;
  unsigned long long int l;

  fa = fabs(a);
  if (fa >= 1.0) {
    l = 0xfff8000000000000ull;
    memcpy(&t, &l, sizeof(double)); /* INDEFINITE */
    if (fa == 1.0) {
      t = a * exp(1000.0); /* Infinity */
    }
  } else if (fa >= 0.9375) {
    /* Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
       Approximations for the Inverse of the Error Function. Mathematics of
       Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 59
     */
    t = log1p(-fa);
    t = 1.0 / sqrt(-t);
    p = 2.7834010353747001060e-3;
    p = p * t + 8.6030097526280260580e-1;
    p = p * t + 2.1371214997265515515e+0;
    p = p * t + 3.1598519601132090206e+0;
    p = p * t + 3.5780402569085996758e+0;
    p = p * t + 1.5335297523989890804e+0;
    p = p * t + 3.4839207139657522572e-1;
    p = p * t + 5.3644861147153648366e-2;
    p = p * t + 4.3836709877126095665e-3;
    p = p * t + 1.3858518113496718808e-4;
    p = p * t + 1.1738352509991666680e-6;
    q = t + 2.2859981272422905412e+0;
    q = q * t + 4.3859045256449554654e+0;
    q = q * t + 4.6632960348736635331e+0;
    q = q * t + 3.9846608184671757296e+0;
    q = q * t + 1.6068377709719017609e+0;
    q = q * t + 3.5609087305900265560e-1;
    q = q * t + 5.3963550303200816744e-2;
    q = q * t + 4.3873424022706935023e-3;
    q = q * t + 1.3858762165532246059e-4;
    q = q * t + 1.1738313872397777529e-6;
    t = p / (q * t);
    if (a < 0.0) t = -t;
  } else if (fa >= 0.75) {
    /* Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
       Approximations for the Inverse of the Error Function. Mathematics of
       Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 39
    */
    t = a * a - .87890625;
    p = .21489185007307062000e+0;
    p = p * t - .64200071507209448655e+1;
    p = p * t + .29631331505876308123e+2;
    p = p * t - .47644367129787181803e+2;
    p = p * t + .34810057749357500873e+2;
    p = p * t - .12954198980646771502e+2;
    p = p * t + .25349389220714893917e+1;
    p = p * t - .24758242362823355486e+0;
    p = p * t + .94897362808681080020e-2;
    q = t - .12831383833953226499e+2;
    q = q * t + .41409991778428888716e+2;
    q = q * t - .53715373448862143349e+2;
    q = q * t + .33880176779595142685e+2;
    q = q * t - .11315360624238054876e+2;
    q = q * t + .20369295047216351160e+1;
    q = q * t - .18611650627372178511e+0;
    q = q * t + .67544512778850945940e-2;
    p = p / q;
    t = a * p;
  } else {
    /* Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
       Approximations for the Inverse of the Error Function. Mathematics of
       Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 18
    */
    t = a * a - .5625;
    p = -.23886240104308755900e+2;
    p = p * t + .45560204272689128170e+3;
    p = p * t - .22977467176607144887e+4;
    p = p * t + .46631433533434331287e+4;
    p = p * t - .43799652308386926161e+4;
    p = p * t + .19007153590528134753e+4;
    p = p * t - .30786872642313695280e+3;
    q = t - .83288327901936570000e+2;
    q = q * t + .92741319160935318800e+3;
    q = q * t - .35088976383877264098e+4;
    q = q * t + .59039348134843665626e+4;
    q = q * t - .48481635430048872102e+4;
    q = q * t + .18997769186453057810e+4;
    q = q * t - .28386514725366621129e+3;
    p = p / q;
    t = a * p;
  }
  return t;
}

const int RandUniformTask::TASK_ID =
    static_cast<int>(NumPyOpCode::NUMPY_RAND_UNIFORM) * NUMPY_TYPE_OFFSET + DOUBLE_LT * NUMPY_MAX_VARIANTS;

/*static*/ void RandUniformTask::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                             Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const unsigned     epoch = derez.unpack_32bit_uint();
  const int          dim   = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const Point<1>              strides = derez.unpack_point<1>();
      const AccessorWO<double, 1> out     = derez.unpack_accessor_WO<double, 1>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
        const unsigned long long key = x * strides[0];
        out[x]                       = RandomGenerator::rand_double(epoch, HI_BITS(key), LO_BITS(key));
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const Point<2>              strides = derez.unpack_point<2>();
      const AccessorWO<double, 2> out     = derez.unpack_accessor_WO<double, 2>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
          const unsigned long long key = x * strides[0] + y * strides[1];
          out[x][y]                    = RandomGenerator::rand_double(epoch, HI_BITS(key), LO_BITS(key));
        }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const Point<3>              strides = derez.unpack_point<3>();
      const AccessorWO<double, 3> out     = derez.unpack_accessor_WO<double, 3>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
            const unsigned long long key = x * strides[0] + y * strides[1] + z * strides[2];
            out[x][y][z]                 = RandomGenerator::rand_double(epoch, HI_BITS(key), LO_BITS(key));
          }
      break;
    }
    default:
      assert(false);
  }
}

#ifdef LEGATE_USE_OPENMP
/*static*/ void RandUniformTask::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                             Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const unsigned     epoch = derez.unpack_32bit_uint();
  const int          dim   = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const Point<1>              strides = derez.unpack_point<1>();
      const AccessorWO<double, 1> out     = derez.unpack_accessor_WO<double, 1>(regions[0], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
        const unsigned long long key = x * strides[0];
        out[x]                       = RandomGenerator::rand_double(epoch, HI_BITS(key), LO_BITS(key));
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const Point<2>              strides = derez.unpack_point<2>();
      const AccessorWO<double, 2> out     = derez.unpack_accessor_WO<double, 2>(regions[0], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
          const unsigned long long key = x * strides[0] + y * strides[1];
          out[x][y]                    = RandomGenerator::rand_double(epoch, HI_BITS(key), LO_BITS(key));
        }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const Point<3>              strides = derez.unpack_point<3>();
      const AccessorWO<double, 3> out     = derez.unpack_accessor_WO<double, 3>(regions[0], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
            const unsigned long long key = x * strides[0] + y * strides[1] + z * strides[2];
            out[x][y][z]                 = RandomGenerator::rand_double(epoch, HI_BITS(key), LO_BITS(key));
          }
      break;
    }
    default:
      assert(false);
  }
}
#endif

const int RandNormalTask::TASK_ID =
    static_cast<int>(NumPyOpCode::NUMPY_RAND_NORMAL) * NUMPY_TYPE_OFFSET + DOUBLE_LT * NUMPY_MAX_VARIANTS;

/*static*/ void RandNormalTask::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                            Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const unsigned     epoch = derez.unpack_32bit_uint();
  const int          dim   = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const Point<1>              strides = derez.unpack_point<1>();
      const AccessorWO<double, 1> out     = derez.unpack_accessor_WO<double, 1>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
        const unsigned long long key = x * strides[0];
        out[x]                       = erfinv(2.0 * RandomGenerator::rand_double(epoch, HI_BITS(key), LO_BITS(key)) - 1.0);
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const Point<2>              strides = derez.unpack_point<2>();
      const AccessorWO<double, 2> out     = derez.unpack_accessor_WO<double, 2>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
          const unsigned long long key = x * strides[0] + y * strides[1];
          out[x][y]                    = erfinv(2.0 * RandomGenerator::rand_double(epoch, HI_BITS(key), LO_BITS(key)) - 1.0);
        }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const Point<3>              strides = derez.unpack_point<3>();
      const AccessorWO<double, 3> out     = derez.unpack_accessor_WO<double, 3>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
            const unsigned long long key = x * strides[0] + y * strides[1] + z * strides[2];
            out[x][y][z]                 = erfinv(2.0 * RandomGenerator::rand_double(epoch, HI_BITS(key), LO_BITS(key)) - 1.0);
          }
      break;
    }
    default:
      assert(false);
  }
}

#ifdef LEGATE_USE_OPENMP
/*static*/ void RandNormalTask::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                            Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const unsigned     epoch = derez.unpack_32bit_uint();
  const int          dim   = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const Point<1>              strides = derez.unpack_point<1>();
      const AccessorWO<double, 1> out     = derez.unpack_accessor_WO<double, 1>(regions[0], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
        const unsigned long long key = x * strides[0];
        out[x]                       = erfinv(2.0 * RandomGenerator::rand_double(epoch, HI_BITS(key), LO_BITS(key)) - 1.0);
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const Point<2>              strides = derez.unpack_point<2>();
      const AccessorWO<double, 2> out     = derez.unpack_accessor_WO<double, 2>(regions[0], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
          const unsigned long long key = x * strides[0] + y * strides[1];
          out[x][y]                    = erfinv(2.0 * RandomGenerator::rand_double(epoch, HI_BITS(key), LO_BITS(key)) - 1.0);
        }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const Point<3>              strides = derez.unpack_point<3>();
      const AccessorWO<double, 3> out     = derez.unpack_accessor_WO<double, 3>(regions[0], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
            const unsigned long long key = x * strides[0] + y * strides[1] + z * strides[2];
            out[x][y][z]                 = erfinv(2.0 * RandomGenerator::rand_double(epoch, HI_BITS(key), LO_BITS(key)) - 1.0);
          }
      break;
    }
    default:
      assert(false);
  }
}
#endif

template<typename T>
/*static*/ void RandIntegerTask<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const unsigned     epoch = derez.unpack_32bit_uint();
  const T            low   = derez.unpack_value<T>();
  const T            high  = derez.unpack_value<T>();
  assert(low < high);
  const unsigned long long diff = high - low;
  const int                dim  = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const Point<1>         strides = derez.unpack_point<1>();
      const AccessorWO<T, 1> out     = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
        const unsigned long long key = x * strides[0];
        out[x]                       = low + RandomGenerator::rand_long(epoch, HI_BITS(key), LO_BITS(key), diff);
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const Point<2>         strides = derez.unpack_point<2>();
      const AccessorWO<T, 2> out     = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
          const unsigned long long key = x * strides[0] + y * strides[1];
          out[x][y]                    = low + RandomGenerator::rand_long(epoch, HI_BITS(key), LO_BITS(key), diff);
        }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const Point<3>         strides = derez.unpack_point<3>();
      const AccessorWO<T, 3> out     = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
            const unsigned long long key = x * strides[0] + y * strides[1] + z * strides[2];
            out[x][y][z]                 = low + RandomGenerator::rand_long(epoch, HI_BITS(key), LO_BITS(key), diff);
          }
      break;
    }
    default:
      assert(false);
  }
}

#ifdef LEGATE_USE_OPENMP
template<typename T>
/*static*/ void RandIntegerTask<T>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const unsigned     epoch = derez.unpack_32bit_uint();
  const T            low   = derez.unpack_value<T>();
  const T            high  = derez.unpack_value<T>();
  assert(low < high);
  const unsigned long long diff = high - low;
  const int                dim  = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const Point<1>         strides = derez.unpack_point<1>();
      const AccessorWO<T, 1> out     = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
        const unsigned long long key = x * strides[0];
        out[x]                       = low + RandomGenerator::rand_long(epoch, HI_BITS(key), LO_BITS(key), diff);
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const Point<2>         strides = derez.unpack_point<2>();
      const AccessorWO<T, 2> out     = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++) {
          const unsigned long long key = x * strides[0] + y * strides[1];
          out[x][y]                    = low + RandomGenerator::rand_long(epoch, HI_BITS(key), LO_BITS(key), diff);
        }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const Point<3>         strides = derez.unpack_point<3>();
      const AccessorWO<T, 3> out     = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
#  pragma omp parallel for
      for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
        for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
          for (coord_t z = rect.lo[2]; z <= rect.hi[2]; z++) {
            const unsigned long long key = x * strides[0] + y * strides[1] + z * strides[2];
            out[x][y][z]                 = low + RandomGenerator::rand_long(epoch, HI_BITS(key), LO_BITS(key), diff);
          }
      break;
    }
    default:
      assert(false);
  }
}

#endif

// No need to instantiate floating point versions

INSTANTIATE_INT_TASKS(RandIntegerTask, static_cast<int>(NumPyOpCode::NUMPY_RAND_INTEGER) * NUMPY_TYPE_OFFSET)

}    // namespace numpy
}    // namespace legate

namespace    // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  legate::numpy::RandUniformTask::register_variants();
  legate::numpy::RandNormalTask::register_variants();
  REGISTER_INT_TASKS(legate::numpy::RandIntegerTask)
}
}    // namespace
