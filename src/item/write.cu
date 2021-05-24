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

#include "item/write.h"
#include "item/write_template.inl"

namespace legate {
namespace numpy {

using namespace Legion;

template <typename VAL, int DIM>
static __global__ void __launch_bounds__(1, 1)
  write_value(const AccessorWO<VAL, DIM> out, const Point<DIM> key, VAL value)
{
  out[key] = value;
}

template <typename VAL, int DIM>
struct WriteImplBody<VariantKind::GPU, VAL, DIM> {
  void operator()(AccessorWO<VAL, DIM> out, const Point<DIM> &key, const VAL &value) const
  {
    write_value<VAL, DIM><<<1, 1>>>(out, key, value);
  }
};

/*static*/ void WriteTask::gpu_variant(const Task *task,
                                       const std::vector<PhysicalRegion> &regions,
                                       Context context,
                                       Runtime *runtime)
{
  write_template<VariantKind::GPU>(task, regions, context, runtime);
}

}  // namespace numpy
}  // namespace legate
