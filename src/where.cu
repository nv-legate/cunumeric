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
#include "where.h"

using namespace Legion;

namespace legate {
namespace numpy {

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_where_1d(const AccessorWO<T, 1> out,
                  const AccessorRO<bool, 1> cond,
                  const AccessorRO<T, 1> in1,
                  const AccessorRO<T, 1> in2,
                  const Point<1> origin,
                  const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset;
  out[x]          = cond[x] ? in1[x] : in2[x];
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_where_2d(const AccessorWO<T, 2> out,
                  const AccessorRO<bool, 2> cond,
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
  out[x][y]       = cond[x][y] ? in1[x][y] : in2[x][y];
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_where_3d(const AccessorWO<T, 3> out,
                  const AccessorRO<bool, 3> cond,
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
  out[x][y][z]    = cond[x][y][z] ? in1[x][y][z] : in2[x][y][z];
}

template <typename T>
/*static*/ void WhereTask<T>::gpu_variant(const Task* task,
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
      const AccessorWO<T, 1> out     = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
      const AccessorRO<bool, 1> cond = derez.unpack_accessor_RO<bool, 1>(regions[1], rect);
      const AccessorRO<T, 1> in1     = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
      const AccessorRO<T, 1> in2     = derez.unpack_accessor_RO<T, 1>(regions[3], rect);
      const size_t volume            = rect.volume();
      const size_t blocks            = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      legate_where_1d<T><<<blocks, THREADS_PER_BLOCK>>>(out, cond, in1, in2, rect.lo, volume);
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 2> out     = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
      const AccessorRO<bool, 2> cond = derez.unpack_accessor_RO<bool, 2>(regions[1], rect);
      const AccessorRO<T, 2> in1     = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
      const AccessorRO<T, 2> in2     = derez.unpack_accessor_RO<T, 2>(regions[3], rect);
      const size_t volume            = rect.volume();
      const size_t blocks            = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t pitch            = rect.hi[1] - rect.lo[1] + 1;
      legate_where_2d<T>
        <<<blocks, THREADS_PER_BLOCK>>>(out, cond, in1, in2, rect.lo, Point<1>(pitch), volume);
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const AccessorWO<T, 3> out     = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
      const AccessorRO<bool, 3> cond = derez.unpack_accessor_RO<bool, 3>(regions[1], rect);
      const AccessorRO<T, 3> in1     = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
      const AccessorRO<T, 3> in2     = derez.unpack_accessor_RO<T, 3>(regions[3], rect);
      const size_t volume            = rect.volume();
      const size_t blocks            = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t diffy            = rect.hi[1] - rect.lo[1] + 1;
      const coord_t diffz            = rect.hi[2] - rect.lo[2] + 1;
      const coord_t pitch[2]         = {diffy * diffz, diffz};
      legate_where_3d<T>
        <<<blocks, THREADS_PER_BLOCK>>>(out, cond, in1, in2, rect.lo, Point<2>(pitch), volume);
      break;
    }
    default: assert(false);
  }
}

INSTANTIATE_TASK_VARIANT(WhereTask, gpu_variant)

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_where_future_1d(const AccessorWO<T, 1> out,
                         const AccessorRO<bool, 1> cond,
                         const T in1,
                         const T in2,
                         const Point<1> origin,
                         const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset;
  out[x]          = cond[x] ? in1 : in2;
}

template <typename T, bool FIRST>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_where_broadcast_1d(const AccessorWO<T, 1> out,
                            const AccessorRO<bool, 1> cond,
                            const T future,
                            const AccessorRO<T, 1> in,
                            const Point<1> origin,
                            const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset;
  if (FIRST)
    out[x] = cond[x] ? future : in[x];
  else
    out[x] = cond[x] ? in[x] : future;
}

template <typename T, bool FIRST>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_where_set_1d(const AccessorWO<T, 1> out,
                      const T future,
                      const AccessorRO<T, 1> in,
                      const Point<1> origin,
                      const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset;
  if (FIRST)
    out[x] = future;
  else
    out[x] = in[x];
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_where_future_2d(const AccessorWO<T, 2> out,
                         const AccessorRO<bool, 2> cond,
                         const T in1,
                         const T in2,
                         const Point<2> origin,
                         const Point<1> pitch,
                         const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + offset % pitch[0];
  out[x][y]       = cond[x][y] ? in1 : in2;
}

template <typename T, bool FIRST>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_where_broadcast_2d(const AccessorWO<T, 2> out,
                            const AccessorRO<bool, 2> cond,
                            const T future,
                            const AccessorRO<T, 2> in,
                            const Point<2> origin,
                            const Point<1> pitch,
                            const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + offset % pitch[0];
  if (FIRST)
    out[x][y] = cond[x][y] ? future : in[x][y];
  else
    out[x][y] = cond[x][y] ? in[x][y] : future;
}

template <typename T, bool FIRST>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_where_set_2d(const AccessorWO<T, 2> out,
                      const T future,
                      const AccessorRO<T, 2> in,
                      const Point<2> origin,
                      const Point<1> pitch,
                      const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + offset % pitch[0];
  if (FIRST)
    out[x][y] = future;
  else
    out[x][y] = in[x][y];
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_where_future_3d(const AccessorWO<T, 3> out,
                         const AccessorRO<bool, 3> cond,
                         const T in1,
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
  out[x][y][z]    = cond[x][y][z] ? in1 : in2;
}

template <typename T, bool FIRST>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_where_broadcast_3d(const AccessorWO<T, 3> out,
                            const AccessorRO<bool, 3> cond,
                            const T future,
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
  if (FIRST)
    out[x][y][z] = cond[x][y][z] ? future : in[x][y][z];
  else
    out[x][y][z] = cond[x][y][z] ? in[x][y][z] : future;
}

template <typename T, bool FIRST>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_where_set_3d(const AccessorWO<T, 3> out,
                      const T future,
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
  if (FIRST)
    out[x][y][z] = future;
  else
    out[x][y][z] = in[x][y][z];
}

template <typename T>
/*static*/ void WhereBroadcast<T>::gpu_variant(const Task* task,
                                               const std::vector<PhysicalRegion>& regions,
                                               Context ctx,
                                               Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int dim         = derez.unpack_dimension();
  unsigned future_index = 0;
  switch (dim) {
    case 1: {
      const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (rect.empty()) break;
      const size_t volume        = rect.volume();
      const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const AccessorWO<T, 1> out = derez.unpack_accessor_WO<T, 1>(regions[0], rect);
      const bool future_cond     = derez.unpack_bool();
      if (future_cond) {
        const bool condition = task->futures[future_index++].get_result<bool>();
        const bool future1   = derez.unpack_bool();
        if (future1) {
          const T in1        = task->futures[future_index++].get_result<T>();
          const bool future2 = derez.unpack_bool();
          assert(!future2);  // this would have been the scalar case
          const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
          if (condition) {
            legate_where_set_1d<T, true>
              <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, volume);
          } else {
            legate_where_set_1d<T, false>
              <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, volume);
          }
        } else {
          const AccessorRO<T, 1> in1 = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
          const bool future2         = derez.unpack_bool();
          if (future2) {
            const T in2 = task->futures[future_index++].get_result<T>();
            if (condition) {
              legate_where_set_1d<T, false>
                <<<blocks, THREADS_PER_BLOCK>>>(out, in2, in1, rect.lo, volume);
            } else {
              legate_where_set_1d<T, true>
                <<<blocks, THREADS_PER_BLOCK>>>(out, in2, in1, rect.lo, volume);
            }
          } else {
            const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
            if (condition) {
              legate_where_set_1d<T, false>
                <<<blocks, THREADS_PER_BLOCK>>>(out, 0, in1, rect.lo, volume);
            } else {
              legate_where_set_1d<T, false>
                <<<blocks, THREADS_PER_BLOCK>>>(out, 0, in2, rect.lo, volume);
            }
          }
        }
      } else {
        const AccessorRO<bool, 1> cond = derez.unpack_accessor_RO<bool, 1>(regions[1], rect);
        const bool future1             = derez.unpack_bool();
        if (future1) {
          const T in1        = task->futures[future_index++].get_result<T>();
          const bool future2 = derez.unpack_bool();
          if (future2) {
            const T in2 = task->futures[future_index++].get_result<T>();
            legate_where_future_1d<T>
              <<<blocks, THREADS_PER_BLOCK>>>(out, cond, in1, in2, rect.lo, volume);
          } else {
            const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
            legate_where_broadcast_1d<T, true>
              <<<blocks, THREADS_PER_BLOCK>>>(out, cond, in1, in2, rect.lo, volume);
          }
        } else {
          const AccessorRO<T, 1> in1 = derez.unpack_accessor_RO<T, 1>(regions[2], rect);
          const bool future2         = derez.unpack_bool();
          assert(future2);  // this would have been the general case
          const T in2 = task->futures[future_index++].get_result<T>();
          legate_where_broadcast_1d<T, false>
            <<<blocks, THREADS_PER_BLOCK>>>(out, cond, in2, in1, rect.lo, volume);
        }
      }
      break;
    }
    case 2: {
      const Rect<2> rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (rect.empty()) break;
      const size_t volume        = rect.volume();
      const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t pitch        = rect.hi[1] - rect.lo[1] + 1;
      const AccessorWO<T, 2> out = derez.unpack_accessor_WO<T, 2>(regions[0], rect);
      const bool future_cond     = derez.unpack_bool();
      if (future_cond) {
        const bool condition = task->futures[future_index++].get_result<bool>();
        const bool future1   = derez.unpack_bool();
        if (future1) {
          const T in1        = task->futures[future_index++].get_result<T>();
          const bool future2 = derez.unpack_bool();
          assert(!future2);  // this would have been the scalar case
          const AccessorRO<T, 2> in2 = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
          if (condition) {
            legate_where_set_2d<T, true>
              <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, Point<1>(pitch), volume);
          } else {
            legate_where_set_2d<T, false>
              <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, Point<1>(pitch), volume);
          }
        } else {
          const AccessorRO<T, 2> in1 = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
          const bool future2         = derez.unpack_bool();
          if (future2) {
            const T in2 = task->futures[future_index++].get_result<T>();
            if (condition) {
              legate_where_set_2d<T, false>
                <<<blocks, THREADS_PER_BLOCK>>>(out, in2, in1, rect.lo, Point<1>(pitch), volume);
            } else {
              legate_where_set_2d<T, true>
                <<<blocks, THREADS_PER_BLOCK>>>(out, in2, in1, rect.lo, Point<1>(pitch), volume);
            }
          } else {
            const AccessorRO<T, 2> in2 = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
            if (condition) {
              legate_where_set_2d<T, false>
                <<<blocks, THREADS_PER_BLOCK>>>(out, 0, in1, rect.lo, Point<1>(pitch), volume);
            } else {
              legate_where_set_2d<T, false>
                <<<blocks, THREADS_PER_BLOCK>>>(out, 0, in2, rect.lo, Point<1>(pitch), volume);
            }
          }
        }
      } else {
        const AccessorRO<bool, 2> cond = derez.unpack_accessor_RO<bool, 2>(regions[1], rect);
        const bool future1             = derez.unpack_bool();
        if (future1) {
          const T in1        = task->futures[future_index++].get_result<T>();
          const bool future2 = derez.unpack_bool();
          if (future2) {
            const T in2 = task->futures[future_index++].get_result<T>();
            legate_where_future_2d<T><<<blocks, THREADS_PER_BLOCK>>>(
              out, cond, in1, in2, rect.lo, Point<1>(pitch), volume);
          } else {
            const AccessorRO<T, 2> in2 = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
            legate_where_broadcast_2d<T, true><<<blocks, THREADS_PER_BLOCK>>>(
              out, cond, in1, in2, rect.lo, Point<1>(pitch), volume);
          }
        } else {
          const AccessorRO<T, 2> in1 = derez.unpack_accessor_RO<T, 2>(regions[2], rect);
          const bool future2         = derez.unpack_bool();
          assert(future2);  // this would have been the general case
          const T in2 = task->futures[future_index++].get_result<T>();
          legate_where_broadcast_2d<T, false>
            <<<blocks, THREADS_PER_BLOCK>>>(out, cond, in2, in1, rect.lo, Point<1>(pitch), volume);
        }
      }
      break;
    }
    case 3: {
      const Rect<3> rect = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      if (rect.empty()) break;
      const size_t volume        = rect.volume();
      const size_t blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      const coord_t diffy        = rect.hi[1] - rect.lo[1] + 1;
      const coord_t diffz        = rect.hi[2] - rect.lo[2] + 1;
      const coord_t pitch[2]     = {diffy * diffz, diffz};
      const AccessorWO<T, 3> out = derez.unpack_accessor_WO<T, 3>(regions[0], rect);
      const bool future_cond     = derez.unpack_bool();
      if (future_cond) {
        const bool condition = task->futures[future_index++].get_result<bool>();
        const bool future1   = derez.unpack_bool();
        if (future1) {
          const T in1        = task->futures[future_index++].get_result<T>();
          const bool future2 = derez.unpack_bool();
          assert(!future2);  // this would have been the scalar case
          const AccessorRO<T, 3> in2 = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
          if (condition) {
            legate_where_set_3d<T, true>
              <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, Point<2>(pitch), volume);
          } else {
            legate_where_set_3d<T, false>
              <<<blocks, THREADS_PER_BLOCK>>>(out, in1, in2, rect.lo, Point<2>(pitch), volume);
          }
        } else {
          const AccessorRO<T, 3> in1 = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
          const bool future2         = derez.unpack_bool();
          if (future2) {
            const T in2 = task->futures[future_index++].get_result<T>();
            if (condition) {
              legate_where_set_3d<T, false>
                <<<blocks, THREADS_PER_BLOCK>>>(out, in2, in1, rect.lo, Point<2>(pitch), volume);
            } else {
              legate_where_set_3d<T, true>
                <<<blocks, THREADS_PER_BLOCK>>>(out, in2, in1, rect.lo, Point<2>(pitch), volume);
            }
          } else {
            const AccessorRO<T, 3> in2 = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
            if (condition) {
              legate_where_set_3d<T, false>
                <<<blocks, THREADS_PER_BLOCK>>>(out, 0, in1, rect.lo, Point<2>(pitch), volume);
            } else {
              legate_where_set_3d<T, false>
                <<<blocks, THREADS_PER_BLOCK>>>(out, 0, in2, rect.lo, Point<2>(pitch), volume);
            }
          }
        }
      } else {
        const AccessorRO<bool, 3> cond = derez.unpack_accessor_RO<bool, 3>(regions[1], rect);
        const bool future1             = derez.unpack_bool();
        if (future1) {
          const T in1        = task->futures[future_index++].get_result<T>();
          const bool future2 = derez.unpack_bool();
          if (future2) {
            const T in2 = task->futures[future_index++].get_result<T>();
            legate_where_future_3d<T><<<blocks, THREADS_PER_BLOCK>>>(
              out, cond, in1, in2, rect.lo, Point<2>(pitch), volume);
          } else {
            const AccessorRO<T, 3> in2 = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
            legate_where_broadcast_3d<T, true><<<blocks, THREADS_PER_BLOCK>>>(
              out, cond, in1, in2, rect.lo, Point<2>(pitch), volume);
          }
        } else {
          const AccessorRO<T, 3> in1 = derez.unpack_accessor_RO<T, 3>(regions[2], rect);
          const bool future2         = derez.unpack_bool();
          assert(future2);  // this would have been the general case
          const T in2 = task->futures[future_index++].get_result<T>();
          legate_where_broadcast_3d<T, false>
            <<<blocks, THREADS_PER_BLOCK>>>(out, cond, in2, in1, rect.lo, Point<2>(pitch), volume);
        }
      }
      break;
    }
    default: assert(false);
  }
}

INSTANTIATE_TASK_VARIANT(WhereBroadcast, gpu_variant)

}  // namespace numpy
}  // namespace legate
