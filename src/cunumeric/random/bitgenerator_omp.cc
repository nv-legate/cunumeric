/* Copyright 2021-2022 NVIDIA Corporation
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

#include "cunumeric/random/bitgenerator.h"
#include "cunumeric/random/bitgenerator_template.inl"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <>
struct BitGeneratorImplBody<VariantKind::OMP> {
  void operator()(BitGeneratorOperation op,
                  int32_t generatorID,
                  uint64_t parameter,
                  const DomainPoint& strides,
                  std::vector<legate::Store>& output,
                  std::vector<legate::Store>& args)
  {
    printf("[INFO] : @ %s : %d\n", __FILE__, __LINE__);
    assert(false);
  }
};

/*static*/ void BitGeneratorTask::omp_variant(legate::TaskContext& context)
{
  bitgenerator_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric