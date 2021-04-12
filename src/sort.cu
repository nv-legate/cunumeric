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
#include "sort.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

using namespace Legion;

namespace legate {
namespace numpy {
template<typename T>
/*static*/ void SortTask<T>::gpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                         Runtime* runtime) {
  LegateDeserializer     derez(task->args, task->arglen);
  const Rect<1>          rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  const AccessorRW<T, 1> out  = derez.unpack_accessor_RW<T, 1>(regions[0], rect);
  if (rect.empty()) return;
  T*           ptr    = out.ptr(rect);
  const size_t volume = rect.volume();
  // Call GPU sort using Thrust
  thrust::device_ptr<T> ptr_d(ptr);
  thrust::sort(ptr_d, ptr_d + volume);
}

INSTANTIATE_TASK_VARIANT(SortTask, gpu_variant)
}    // namespace numpy
}    // namespace legate
