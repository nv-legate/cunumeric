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
#include "mod.h"
#include "proj.h"

using namespace Legion;

namespace legate {
namespace numpy {

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_int_mod_1d(const AccessorWO<T, 1> out,
                    const AccessorRO<T, 1> in1,
                    const AccessorRO<T, 1> in2,
                    const Point<1> origin,
                    const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset;
  out[x]          = in1[x] % in2[x];
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) legate_int_mod_1d_inplace(
  const AccessorRW<T, 1> out, const AccessorRO<T, 1> in, const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset;
  out[x] %= in[x];
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_int_mod_2d(const AccessorWO<T, 2> out,
                    const AccessorRO<T, 2> in1,
                    const AccessorRO<T, 2> in2,
                    const Point<2> origin,
                    const Point<1> pitch,
                    const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + offset % pitch[0];
  out[x][y]       = in1[x][y] % in2[x][y];
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_int_mod_2d_inplace(const AccessorRW<T, 2> out,
                            const AccessorRO<T, 2> in,
                            const Point<2> origin,
                            const Point<1> pitch,
                            const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + offset % pitch[0];
  out[x][y] %= in[x][y];
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_int_mod_3d(const AccessorWO<T, 3> out,
                    const AccessorRO<T, 3> in1,
                    const AccessorRO<T, 3> in2,
                    const Point<3> origin,
                    const Point<2> pitch,
                    const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z = origin[2] + (offset % pitch[0]) % pitch[1];
  out[x][y][z]    = in1[x][y][z] % in2[x][y][z];
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_int_mod_3d_inplace(const AccessorRW<T, 3> out,
                            const AccessorRO<T, 3> in,
                            const Point<3> origin,
                            const Point<2> pitch,
                            const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z = origin[2] + (offset % pitch[0]) % pitch[1];
  out[x][y][z] %= in[x][y][z];
}

template <typename T>
/*static*/ void IntModTask<T>::gpu_variant(const Task* task,
                                           const std::vector<PhysicalRegion>& regions,
                                           Context ctx,
                                           Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int dim = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 2) {
        const AccessorRW<T, 1> out = derez.unpack_accessor_RW<T, 1>(regions[0], rect);
        const AccessorRO<T, 1> in  = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
        const size_t volume        = rect.volume();
        const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        legate_int_mod_1d_inplace<T><<<blocks, THREADS_PER_BLOCK>>>(out, in, rect.lo, volume);
      } else {
        const AccessorWO<T, 1> out = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
        const AccessorRO<T, 1> in1 = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
        const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
        const size_t volume        = rect.volume();
        const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        legate_int_mod_1d<T><<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, volume);
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 2) {
        const AccessorRW<T, 2> out = derez.unpack_accessor_RW<T, 2>(regions[0], rect);
        const AccessorRO<T, 2> in  = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
        const size_t volume        = rect.volume();
        const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const coord_t pitch        = rect.hi[1] - rect.lo[1] + 1;
        legate_int_mod_2d_inplace<T>
          <<<blocks, THREADS_PER_BLOCK>>>(out, in, rect.lo, Point<1>(pitch), volume);
      } else {
        const AccessorWO<T, 2> out = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
        const AccessorRO<T, 2> in1 = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
        const AccessorRO<T, 2> in2 = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
        const size_t volume        = rect.volume();
        const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const coord_t pitch        = rect.hi[1] - rect.lo[1] + 1;
        legate_int_mod_2d<T>
          <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, Point<1>(pitch), volume);
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 2) {
        const AccessorRW<T, 3> out = derez.unpack_accessor_RW<T, 3>(regions[0], rect);
        const AccessorRO<T, 3> in  = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
        const size_t volume        = rect.volume();
        const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const coord_t diffy        = rect.hi[1] - rect.lo[1] + 1;
        const coord_t diffz        = rect.hi[2] - rect.lo[2] + 1;
        const coord_t pitch[2]     = {diffy * diffz, diffz};
        legate_int_mod_3d_inplace<T>
          <<<blocks, THREADS_PER_BLOCK>>>(out, in, rect.lo, Point<2>(pitch), volume);
      } else {
        const AccessorWO<T, 3> out = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
        const AccessorRO<T, 3> in1 = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
        const AccessorRO<T, 3> in2 = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
        const size_t volume        = rect.volume();
        const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const coord_t diffy        = rect.hi[1] - rect.lo[1] + 1;
        const coord_t diffz        = rect.hi[2] - rect.lo[2] + 1;
        const coord_t pitch[2]     = {diffy * diffz, diffz};
        legate_int_mod_3d<T>
          <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, Point<2>(pitch), volume);
      }
      break;
    }
    default: assert(false);
  }
}

INSTANTIATE_INT_VARIANT(IntModTask, gpu_variant)

__device__ __forceinline__ static __half fmod(__half lhs, __half rhs)
{
  return (__half)fmod((float)lhs, (float)rhs);
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_real_mod_1d(const AccessorWO<T, 1> out,
                     const AccessorRO<T, 1> in1,
                     const AccessorRO<T, 1> in2,
                     const Point<1> origin,
                     const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset;
  out[x]          = fmod(in1[x], in2[x]);
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) legate_real_mod_1d_inplace(
  const AccessorRW<T, 1> out, const AccessorRO<T, 1> in, const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset;
  out[x]          = fmod(out[x], in[x]);
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_real_mod_2d(const AccessorWO<T, 2> out,
                     const AccessorRO<T, 2> in1,
                     const AccessorRO<T, 2> in2,
                     const Point<2> origin,
                     const Point<1> pitch,
                     const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + offset % pitch[0];
  out[x][y]       = fmod(in1[x][y], in2[x][y]);
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_real_mod_2d_inplace(const AccessorRW<T, 2> out,
                             const AccessorRO<T, 2> in,
                             const Point<2> origin,
                             const Point<1> pitch,
                             const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + offset % pitch[0];
  out[x][y]       = fmod(out[x][y], in[x][y]);
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_real_mod_3d(const AccessorWO<T, 3> out,
                     const AccessorRO<T, 3> in1,
                     const AccessorRO<T, 3> in2,
                     const Point<3> origin,
                     const Point<2> pitch,
                     const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z = origin[2] + (offset % pitch[0]) % pitch[1];
  out[x][y][z]    = fmod(in1[x][y][z], in2[x][y][z]);
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_real_mod_3d_inplace(const AccessorRW<T, 3> out,
                             const AccessorRO<T, 3> in,
                             const Point<3> origin,
                             const Point<2> pitch,
                             const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z = origin[2] + (offset % pitch[0]) % pitch[1];
  out[x][y][z]    = fmod(out[x][y][z], in[x][y][z]);
}

template <typename T>
/*static*/ void RealModTask<T>::gpu_variant(const Task* task,
                                            const std::vector<PhysicalRegion>& regions,
                                            Context ctx,
                                            Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int dim = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 2) {
        const AccessorRW<T, 1> out = derez.unpack_accessor_RW<T, 1>(regions[0], rect);
        const AccessorRO<T, 1> in  = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
        const size_t volume        = rect.volume();
        const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        legate_real_mod_1d_inplace<T><<<blocks, THREADS_PER_BLOCK>>>(out, in, rect.lo, volume);
      } else {
        const AccessorWO<T, 1> out = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
        const AccessorRO<T, 1> in1 = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
        const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
        const size_t volume        = rect.volume();
        const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        legate_real_mod_1d<T><<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, volume);
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 2) {
        const AccessorRW<T, 2> out = derez.unpack_accessor_RW<T, 2>(regions[0], rect);
        const AccessorRO<T, 2> in  = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
        const size_t volume        = rect.volume();
        const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const coord_t pitch        = rect.hi[1] - rect.lo[1] + 1;
        legate_real_mod_2d_inplace<T>
          <<<blocks, THREADS_PER_BLOCK>>>(out, in, rect.lo, Point<1>(pitch), volume);
      } else {
        const AccessorWO<T, 2> out = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
        const AccessorRO<T, 2> in1 = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
        const AccessorRO<T, 2> in2 = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
        const size_t volume        = rect.volume();
        const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const coord_t pitch        = rect.hi[1] - rect.lo[1] + 1;
        legate_real_mod_2d<T>
          <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, Point<1>(pitch), volume);
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 2) {
        const AccessorRW<T, 3> out = derez.unpack_accessor_RW<T, 3>(regions[0], rect);
        const AccessorRO<T, 3> in  = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
        const size_t volume        = rect.volume();
        const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const coord_t diffy        = rect.hi[1] - rect.lo[1] + 1;
        const coord_t diffz        = rect.hi[2] - rect.lo[2] + 1;
        const coord_t pitch[2]     = {diffy * diffz, diffz};
        legate_real_mod_3d_inplace<T>
          <<<blocks, THREADS_PER_BLOCK>>>(out, in, rect.lo, Point<2>(pitch), volume);
      } else {
        const AccessorWO<T, 3> out = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
        const AccessorRO<T, 3> in1 = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
        const AccessorRO<T, 3> in2 = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
        const size_t volume        = rect.volume();
        const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const coord_t diffy        = rect.hi[1] - rect.lo[1] + 1;
        const coord_t diffz        = rect.hi[2] - rect.lo[2] + 1;
        const coord_t pitch[2]     = {diffy * diffz, diffz};
        legate_real_mod_3d<T>
          <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, Point<2>(pitch), volume);
      }
      break;
    }
    default: assert(false);
  }
}

INSTANTIATE_REAL_VARIANT(RealModTask, gpu_variant)

template <typename T, bool FIRST>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_int_mod_broadcast_1d(const AccessorWO<T, 1> out,
                              const AccessorRO<T, 1> in1,
                              const T in2,
                              const Point<1> origin,
                              const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset;
  if (FIRST)
    out[x] = in2 % in1[x];
  else
    out[x] = in1[x] % in2;
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_int_mod_broadcast_1d_inplace(const AccessorRW<T, 1> out,
                                      const T in,
                                      const Point<1> origin,
                                      const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset;
  out[x] %= in;
}

template <typename T, bool FIRST>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_int_mod_broadcast_2d(const AccessorWO<T, 2> out,
                              const AccessorRO<T, 2> in1,
                              const T in2,
                              const Point<2> origin,
                              const Point<1> pitch,
                              const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + offset % pitch[0];
  if (FIRST)
    out[x][y] = in2 % in1[x][y];
  else
    out[x][y] = in1[x][y] % in2;
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_int_mod_broadcast_2d_inplace(const AccessorRW<T, 2> out,
                                      const T in,
                                      const Point<2> origin,
                                      const Point<1> pitch,
                                      const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + offset % pitch[0];
  out[x][y] %= in;
}

template <typename T, bool FIRST>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_int_mod_broadcast_3d(const AccessorWO<T, 3> out,
                              const AccessorRO<T, 3> in1,
                              const T in2,
                              const Point<3> origin,
                              const Point<2> pitch,
                              const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z = origin[2] + (offset % pitch[0]) % pitch[1];
  if (FIRST)
    out[x][y][z] = in2 % in1[x][y][z];
  else
    out[x][y][z] = in1[x][y][z] % in2;
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_int_mod_broadcast_3d_inplace(const AccessorRW<T, 3> out,
                                      const T in,
                                      const Point<3> origin,
                                      const Point<2> pitch,
                                      const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z = origin[2] + (offset % pitch[0]) % pitch[1];
  out[x][y][z] %= in;
}

template <typename T>
/*static*/ void IntModBroadcast<T>::gpu_variant(const Task* task,
                                                const std::vector<PhysicalRegion>& regions,
                                                Context ctx,
                                                Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int dim = derez.unpack_dimension();
  assert(task->futures.size() == 1);
  const T in2 = task->futures[0].get_result<T>();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 1) {
        const AccessorRW<T, 1> out = derez.unpack_accessor_RW<T, 1>(regions[0], rect);
        const size_t volume        = rect.volume();
        const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        legate_int_mod_broadcast_1d_inplace<T>
          <<<blocks, THREADS_PER_BLOCK>>>(out, in2, rect.lo, volume);
      } else {
        const AccessorWO<T, 1> out = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
        const AccessorRO<T, 1> in1 = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
        const unsigned index       = derez.unpack_32bit_uint();
        assert((index == 0) || (index == 1));
        const size_t volume = rect.volume();
        const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        if (index == 0)
          legate_int_mod_broadcast_1d<T, true>
            <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, volume);
        else
          legate_int_mod_broadcast_1d<T, false>
            <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, volume);
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 1) {
        const AccessorRW<T, 2> out = derez.unpack_accessor_RW<T, 2>(regions[0], rect);
        const size_t volume        = rect.volume();
        const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const coord_t pitch        = rect.hi[1] - rect.lo[1] + 1;
        legate_int_mod_broadcast_2d_inplace<T>
          <<<blocks, THREADS_PER_BLOCK>>>(out, in2, rect.lo, Point<1>(pitch), volume);
      } else {
        const AccessorWO<T, 2> out = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
        const AccessorRO<T, 2> in1 = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
        const unsigned index       = derez.unpack_32bit_uint();
        assert((index == 0) || (index == 1));
        const size_t volume = rect.volume();
        const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const coord_t pitch = rect.hi[1] - rect.lo[1] + 1;
        if (index == 0)
          legate_int_mod_broadcast_2d<T, true>
            <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, Point<1>(pitch), volume);
        else
          legate_int_mod_broadcast_2d<T, false>
            <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, Point<1>(pitch), volume);
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 1) {
        const AccessorRW<T, 3> out = derez.unpack_accessor_RW<T, 3>(regions[0], rect);
        const size_t volume        = rect.volume();
        const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const coord_t diffy        = rect.hi[1] - rect.lo[1] + 1;
        const coord_t diffz        = rect.hi[2] - rect.lo[2] + 1;
        const coord_t pitch[2]     = {diffy * diffz, diffz};
        legate_int_mod_broadcast_3d_inplace<T>
          <<<blocks, THREADS_PER_BLOCK>>>(out, in2, rect.lo, Point<2>(pitch), volume);
      } else {
        const AccessorWO<T, 3> out = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
        const AccessorRO<T, 3> in1 = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
        const unsigned index       = derez.unpack_32bit_uint();
        assert((index == 0) || (index == 1));
        const size_t volume    = rect.volume();
        const size_t blocks    = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const coord_t diffy    = rect.hi[1] - rect.lo[1] + 1;
        const coord_t diffz    = rect.hi[2] - rect.lo[2] + 1;
        const coord_t pitch[2] = {diffy * diffz, diffz};
        if (index == 0)
          legate_int_mod_broadcast_3d<T, true>
            <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, Point<2>(pitch), volume);
        else
          legate_int_mod_broadcast_3d<T, false>
            <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, Point<2>(pitch), volume);
      }
      break;
    }
    default: assert(false);
  }
}

INSTANTIATE_INT_VARIANT(IntModBroadcast, gpu_variant)

template <typename T, bool FIRST>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_real_mod_broadcast_1d(const AccessorWO<T, 1> out,
                               const AccessorRO<T, 1> in1,
                               const T in2,
                               const Point<1> origin,
                               const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset;
  if (FIRST)
    out[x] = fmod(in2, in1[x]);
  else
    out[x] = fmod(in1[x], in2);
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_real_mod_broadcast_1d_inplace(const AccessorRW<T, 1> out,
                                       const T in,
                                       const Point<1> origin,
                                       const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset;
  out[x]          = fmod(out[x], in);
}

template <typename T, bool FIRST>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_real_mod_broadcast_2d(const AccessorWO<T, 2> out,
                               const AccessorRO<T, 2> in1,
                               const T in2,
                               const Point<2> origin,
                               const Point<1> pitch,
                               const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + offset % pitch[0];
  if (FIRST)
    out[x][y] = fmod(in2, in1[x][y]);
  else
    out[x][y] = fmod(in1[x][y], in2);
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_real_mod_broadcast_2d_inplace(const AccessorRW<T, 2> out,
                                       const T in,
                                       const Point<2> origin,
                                       const Point<1> pitch,
                                       const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + offset % pitch[0];
  out[x][y]       = fmod(out[x][y], in);
}

template <typename T, bool FIRST>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_real_mod_broadcast_3d(const AccessorWO<T, 3> out,
                               const AccessorRO<T, 3> in1,
                               const T in2,
                               const Point<3> origin,
                               const Point<2> pitch,
                               const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z = origin[2] + (offset % pitch[0]) % pitch[1];
  if (FIRST)
    out[x][y][z] = fmod(in2, in1[x][y][z]);
  else
    out[x][y][z] = fmod(in1[x][y][z], in2);
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_real_mod_broadcast_3d_inplace(const AccessorRW<T, 3> out,
                                       const T in,
                                       const Point<3> origin,
                                       const Point<2> pitch,
                                       const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z = origin[2] + (offset % pitch[0]) % pitch[1];
  out[x][y][z]    = fmod(out[x][y][z], in);
}

template <typename T>
/*static*/ void RealModBroadcast<T>::gpu_variant(const Task* task,
                                                 const std::vector<PhysicalRegion>& regions,
                                                 Context ctx,
                                                 Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int dim = derez.unpack_dimension();
  assert(task->futures.size() == 1);
  const T in2 = task->futures[0].get_result<T>();
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 1) {
        const AccessorRW<T, 1> out = derez.unpack_accessor_RW<T, 1>(regions[0], rect);
        const size_t volume        = rect.volume();
        const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        legate_real_mod_broadcast_1d_inplace<T>
          <<<blocks, THREADS_PER_BLOCK>>>(out, in2, rect.lo, volume);
      } else {
        const AccessorWO<T, 1> out = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
        const AccessorRO<T, 1> in1 = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
        const unsigned index       = derez.unpack_32bit_uint();
        assert((index == 0) || (index == 1));
        const size_t volume = rect.volume();
        const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        if (index == 0)
          legate_real_mod_broadcast_1d<T, true>
            <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, volume);
        else
          legate_real_mod_broadcast_1d<T, false>
            <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, volume);
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 1) {
        const AccessorRW<T, 2> out = derez.unpack_accessor_RW<T, 2>(regions[0], rect);
        const size_t volume        = rect.volume();
        const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const coord_t pitch        = rect.hi[1] - rect.lo[1] + 1;
        legate_real_mod_broadcast_2d_inplace<T>
          <<<blocks, THREADS_PER_BLOCK>>>(out, in2, rect.lo, Point<1>(pitch), volume);
      } else {
        const AccessorWO<T, 2> out = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
        const AccessorRO<T, 2> in1 = derez.unpack_accessor_RO<T, 2>(regions[1], rect);
        const unsigned index       = derez.unpack_32bit_uint();
        assert((index == 0) || (index == 1));
        const size_t volume = rect.volume();
        const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const coord_t pitch = rect.hi[1] - rect.lo[1] + 1;
        if (index == 0)
          legate_real_mod_broadcast_2d<T, true>
            <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, Point<1>(pitch), volume);
        else
          legate_real_mod_broadcast_2d<T, false>
            <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, Point<1>(pitch), volume);
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      if (task->regions.size() == 1) {
        const AccessorRW<T, 3> out = derez.unpack_accessor_RW<T, 3>(regions[0], rect);
        const size_t volume        = rect.volume();
        const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const coord_t diffy        = rect.hi[1] - rect.lo[1] + 1;
        const coord_t diffz        = rect.hi[2] - rect.lo[2] + 1;
        const coord_t pitch[2]     = {diffy * diffz, diffz};
        legate_real_mod_broadcast_3d_inplace<T>
          <<<blocks, THREADS_PER_BLOCK>>>(out, in2, rect.lo, Point<2>(pitch), volume);
      } else {
        const AccessorWO<T, 3> out = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
        const AccessorRO<T, 3> in1 = derez.unpack_accessor_RO<T, 3>(regions[1], rect);
        const unsigned index       = derez.unpack_32bit_uint();
        assert((index == 0) || (index == 1));
        const size_t volume    = rect.volume();
        const size_t blocks    = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const coord_t diffy    = rect.hi[1] - rect.lo[1] + 1;
        const coord_t diffz    = rect.hi[2] - rect.lo[2] + 1;
        const coord_t pitch[2] = {diffy * diffz, diffz};
        if (index == 0)
          legate_real_mod_broadcast_3d<T, true>
            <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, Point<2>(pitch), volume);
        else
          legate_real_mod_broadcast_3d<T, false>
            <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, Point<2>(pitch), volume);
      }
      break;
    }
    default: assert(false);
  }
}

INSTANTIATE_REAL_VARIANT(RealModBroadcast, gpu_variant)

}  // namespace numpy
}  // namespace legate
