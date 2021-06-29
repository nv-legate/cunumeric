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

#include "transform.h"

using namespace Legion;

namespace legate {
namespace numpy {

// Instantiate Transform tasks' gpu variants.
#define DIMFUNC(M, N)                                        \
  template void PointTask<TransformTask<M, N>>::gpu_variant( \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
LEGION_FOREACH_NN(DIMFUNC)
#undef DIMFUNC

}  // namespace numpy
}  // namespace legate
