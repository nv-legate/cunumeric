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

#ifndef __NUMPY_FILL_CUH__
#define __NUMPY_FILL_CUH__

#include "cuda_help.h"
#include "legate.h"
#include "numpy.h"

namespace legate {
namespace numpy {

template<typename T>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_fill_1d(const AccessorWO<T, 1> out, const T value, const Legion::Point<1> origin, const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset;
  out[x]          = value;
}

template<typename T>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_fill_2d(const AccessorWO<T, 2> out, const T value, const Legion::Point<2> origin, const Legion::Point<1> pitch,
                   const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + offset % pitch[0];
  out[x][y]       = value;
}

template<typename T>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
    legate_fill_3d(const AccessorWO<T, 3> out, const T value, const Legion::Point<3> origin, const Legion::Point<2> pitch,
                   const size_t max) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  const coord_t x = origin[0] + offset / pitch[0];
  const coord_t y = origin[1] + (offset % pitch[0]) / pitch[1];
  const coord_t z = origin[2] + (offset % pitch[0]) % pitch[1];
  out[x][y][z]    = value;
}

}    // namespace numpy
}    // namespace legate

#endif    // __NUMPY_FILL_CUH__
