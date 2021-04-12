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

#include "cuda_help.h"
#include "proj.h"
#include "rand.h"

using namespace Legion;

namespace legate {
namespace numpy {

__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_uniform_rand_1d(const AccessorWO<double, 1> out, const Point<1> origin, const Point<1> strides, const unsigned epoch,
                           const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t            x   = origin[0] + offset;
  const unsigned long long key = x * strides[0];
  out[x]                       = RandomGenerator::rand_double(epoch, HI_BITS(key), LO_BITS(key));
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_uniform_rand_2d(const AccessorWO<double, 2> out, const Point<2> origin, const Point<1> pitch, const Point<2> strides,
                           const unsigned epoch, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t            x   = origin[0] + offset / pitch[0];
  const coord_t            y   = origin[1] + offset % pitch[0];
  const unsigned long long key = x * strides[0] + y * strides[1];
  out[x][y]                    = RandomGenerator::rand_double(epoch, HI_BITS(key), LO_BITS(key));
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_uniform_rand_3d(const AccessorWO<double, 3> out, const Point<3> origin, const Point<2> pitch, const Point<3> strides,
                           const unsigned epoch, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t            x   = origin[0] + offset / pitch[0];
  const coord_t            y   = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t            z   = origin[2] + (offset % pitch[0]) % pitch[1];
  const unsigned long long key = x * strides[0] + y * strides[1] + z * strides[2];
  out[x][y][z]                 = RandomGenerator::rand_double(epoch, HI_BITS(key), LO_BITS(key));
}

/*static*/ void RandUniformTask::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
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
      const size_t                volume  = rect.volume();
      const size_t                blocks  = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      legate_uniform_rand_1d<<<blocks, THREADS_PER_BLOCK>>>(out, rect.lo, strides, epoch, volume);
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const Point<2>              strides = derez.unpack_point<2>();
      const AccessorWO<double, 2> out     = derez.unpack_accessor_WO<double, 2>(regions[0], rect);
      const size_t                volume  = rect.volume();
      const size_t                blocks  = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t               pitch   = rect.hi[1] - rect.lo[1] + 1;
      legate_uniform_rand_2d<<<blocks, THREADS_PER_BLOCK>>>(out, rect.lo, pitch, strides, epoch, volume);
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const Point<3>              strides  = derez.unpack_point<3>();
      const AccessorWO<double, 3> out      = derez.unpack_accessor_WO<double, 3>(regions[0], rect);
      const size_t                volume   = rect.volume();
      const size_t                blocks   = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t               diffy    = rect.hi[1] - rect.lo[1] + 1;
      const coord_t               diffz    = rect.hi[2] - rect.lo[2] + 1;
      const coord_t               pitch[2] = {diffy * diffz, diffz};
      legate_uniform_rand_3d<<<blocks, THREADS_PER_BLOCK>>>(out, rect.lo, Point<2>(pitch), strides, epoch, volume);
      break;
    }
    default:
      assert(false);
  }
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_normal_rand_1d(const AccessorWO<double, 1> out, const Point<1> origin, const Point<1> strides, const unsigned epoch,
                          const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t            x   = origin[0] + offset;
  const unsigned long long key = x * strides[0];
  out[x]                       = erfinv(2.0 * RandomGenerator::rand_double(epoch, HI_BITS(key), LO_BITS(key)) - 1.0);
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_normal_rand_2d(const AccessorWO<double, 2> out, const Point<2> origin, const Point<1> pitch, const Point<2> strides,
                          const unsigned epoch, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t            x   = origin[0] + offset / pitch[0];
  const coord_t            y   = origin[1] + offset % pitch[0];
  const unsigned long long key = x * strides[0] + y * strides[1];
  out[x][y]                    = erfinv(2.0 * RandomGenerator::rand_double(epoch, HI_BITS(key), LO_BITS(key)) - 1.0);
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_normal_rand_3d(const AccessorWO<double, 3> out, const Point<3> origin, const Point<2> pitch, const Point<3> strides,
                          const unsigned epoch, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t            x   = origin[0] + offset / pitch[0];
  const coord_t            y   = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t            z   = origin[2] + (offset % pitch[0]) % pitch[1];
  const unsigned long long key = x * strides[0] + y * strides[1] + z * strides[2];
  out[x][y][z]                 = erfinv(2.0 * RandomGenerator::rand_double(epoch, HI_BITS(key), LO_BITS(key)) - 1.0);
}

/*static*/ void RandNormalTask::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
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
      const size_t                volume  = rect.volume();
      const size_t                blocks  = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      legate_normal_rand_1d<<<blocks, THREADS_PER_BLOCK>>>(out, rect.lo, strides, epoch, volume);
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const Point<2>              strides = derez.unpack_point<2>();
      const AccessorWO<double, 2> out     = derez.unpack_accessor_WO<double, 2>(regions[0], rect);
      const size_t                volume  = rect.volume();
      const size_t                blocks  = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t               pitch   = rect.hi[1] - rect.lo[1] + 1;
      legate_normal_rand_2d<<<blocks, THREADS_PER_BLOCK>>>(out, rect.lo, pitch, strides, epoch, volume);
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const Point<3>              strides  = derez.unpack_point<3>();
      const AccessorWO<double, 3> out      = derez.unpack_accessor_WO<double, 3>(regions[0], rect);
      const size_t                volume   = rect.volume();
      const size_t                blocks   = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t               diffy    = rect.hi[1] - rect.lo[1] + 1;
      const coord_t               diffz    = rect.hi[2] - rect.lo[2] + 1;
      const coord_t               pitch[2] = {diffy * diffz, diffz};
      legate_normal_rand_3d<<<blocks, THREADS_PER_BLOCK>>>(out, rect.lo, Point<2>(pitch), strides, epoch, volume);
      break;
    }
    default:
      assert(false);
  }
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_integer_rand_1d(const AccessorWO<T, 1> out, const Point<1> origin, const Point<1> strides, const unsigned long long low,
                           const unsigned long long diff, const unsigned epoch, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t            x   = origin[0] + offset;
  const unsigned long long key = x * strides[0];
  out[x]                       = low + RandomGenerator::rand_long(epoch, HI_BITS(key), LO_BITS(key), diff);
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_integer_rand_2d(const AccessorWO<T, 2> out, const Point<2> origin, const Point<1> pitch, const Point<2> strides,
                           const unsigned long long low, const unsigned long long diff, const unsigned epoch, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t            x   = origin[0] + offset / pitch[0];
  const coord_t            y   = origin[1] + offset % pitch[0];
  const unsigned long long key = x * strides[0] + y * strides[1];
  out[x][y]                    = low + RandomGenerator::rand_long(epoch, HI_BITS(key), LO_BITS(key), diff);
}

template<typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_integer_rand_3d(const AccessorWO<T, 3> out, const Point<3> origin, const Point<2> pitch, const Point<3> strides,
                           const unsigned long long low, const unsigned long long diff, const unsigned epoch, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t            x   = origin[0] + offset / pitch[0];
  const coord_t            y   = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t            z   = origin[2] + (offset % pitch[0]) % pitch[1];
  const unsigned long long key = x * strides[0] + y * strides[1] + z * strides[2];
  out[x][y][z]                 = low + RandomGenerator::rand_long(epoch, HI_BITS(key), LO_BITS(key), diff);
}

template<typename T>
/*static*/ void RandIntegerTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
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
      const size_t           volume  = rect.volume();
      const size_t           blocks  = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      legate_integer_rand_1d<T><<<blocks, THREADS_PER_BLOCK>>>(out, rect.lo, strides, low, diff, epoch, volume);
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const Point<2>         strides = derez.unpack_point<2>();
      const AccessorWO<T, 2> out     = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
      const size_t           volume  = rect.volume();
      const size_t           blocks  = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t          pitch   = rect.hi[1] - rect.lo[1] + 1;
      legate_integer_rand_2d<T><<<blocks, THREADS_PER_BLOCK>>>(out, rect.lo, pitch, strides, low, diff, epoch, volume);
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const Point<3>         strides  = derez.unpack_point<3>();
      const AccessorWO<T, 3> out      = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
      const size_t           volume   = rect.volume();
      const size_t           blocks   = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t          diffy    = rect.hi[1] - rect.lo[1] + 1;
      const coord_t          diffz    = rect.hi[2] - rect.lo[2] + 1;
      const coord_t          pitch[2] = {diffy * diffz, diffz};
      legate_integer_rand_3d<T><<<blocks, THREADS_PER_BLOCK>>>(out, rect.lo, Point<2>(pitch), strides, low, diff, epoch, volume);
      break;
    }
    default:
      assert(false);
  }
}

INSTANTIATE_INT_VARIANT(RandIntegerTask, gpu_variant)

}    // namespace numpy
}    // namespace legate
