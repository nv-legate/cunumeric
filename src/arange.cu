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

#include "arange.h"
#include "cuda_help.h"
#include "proj.h"

using namespace Legion;

namespace legate {
namespace numpy {

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_arange(const AccessorWO<T, 1> out,
                const Point<1> lo,
                const size_t max,
                const T start,
                const T stop,
                const T step)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) return;
  Point<1> p(lo + offset);
  out[p] = (T)p[0] * step + start;
}

template <typename T>
/*static*/ void ArangeTask<T>::gpu_variant(const Task* task,
                                           const std::vector<PhysicalRegion>& regions,
                                           Context ctx,
                                           Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  if (rect.empty()) return;
  const AccessorWO<T, 1> out = derez.unpack_accessor_WO<T, 1>(regions[0], rect);

  const T start = task->futures[0].get_result<T>();
  const T stop  = task->futures[1].get_result<T>();
  const T step  = task->futures[2].get_result<T>();

  const Point<1> lo   = rect.lo;
  const size_t size   = rect.volume();
  const size_t blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  legate_arange<T><<<blocks, THREADS_PER_BLOCK>>>(out, lo, size, start, stop, step);
}

INSTANTIATE_TASK_VARIANT(ArangeTask, gpu_variant)

}  // namespace numpy
}  // namespace legate
