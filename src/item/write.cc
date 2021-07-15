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

template <typename VAL>
struct WriteImplBody<VariantKind::CPU, VAL> {
  void operator()(AccessorWO<VAL, 1> out, const AccessorRO<VAL, 1> &value) const
  {
    out[0] = value[0];
  }
};

/*static*/ void WriteTask::cpu_variant(const Task *task,
                                       const std::vector<PhysicalRegion> &regions,
                                       Context context,
                                       Runtime *runtime)
{
  write_template<VariantKind::CPU>(task, regions, context, runtime);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { WriteTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
