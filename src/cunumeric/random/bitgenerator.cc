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
#include "cunumeric/random/bitgenerator_util.h"

#include "cunumeric/random/curand_help.h"

#include "cunumeric/random/bitgenerator_curand.inl"

namespace cunumeric {

using namespace Legion;
using namespace legate;

struct CPUGenerator : public CURANDGenerator {
  CPUGenerator(BitGeneratorType gentype)
  {
    CHECK_CURAND(::curandCreateGeneratorHost(&gen, get_curandRngType(gentype)));
    // offset is initialized by base class
    CHECK_CURAND(::curandSetGeneratorOffset(gen, offset));
    type               = get_curandRngType(gentype);
    supports_skipahead = supportsSkipAhead(type);
    dev_buffer_size    = DEFAULT_DEV_BUFFER_SIZE;
    dev_buffer         = (uint32_t*)::malloc(dev_buffer_size * sizeof(uint32_t));
  }

  virtual ~CPUGenerator()
  {
    // wait for rand jobs and clean-up resources
    std::lock_guard<std::mutex> guard(lock);
    free(dev_buffer);
    CHECK_CURAND(::curandDestroyGenerator(gen));
  }
};

template <>
struct CURANDGeneratorBuilder<VariantKind::CPU> {
  static CURANDGenerator* build(BitGeneratorType gentype) { return new CPUGenerator(gentype); }

  static void destroy(CURANDGenerator* cugenptr) { delete cugenptr; }
};

template <>
std::map<Legion::Processor, std::unique_ptr<generator_map<VariantKind::CPU>>>
  BitGeneratorImplBody<VariantKind::CPU>::m_generators = {};

template <>
std::mutex BitGeneratorImplBody<VariantKind::CPU>::lock_generators = {};

/*static*/ void BitGeneratorTask::cpu_variant(TaskContext& context)
{
  bitgenerator_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks(void)
{
  BitGeneratorTask::register_variants();
}

}  // namespace

}  // namespace cunumeric