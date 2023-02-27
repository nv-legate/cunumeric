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
#include "cunumeric/random/randutil/randutil.h"

#include "cunumeric/random/bitgenerator_curand.inl"

namespace cunumeric {

using namespace legate;

static Logger log_curand("cunumeric.random");

Logger& randutil_log() { return log_curand; }

void randutil_check_curand(curandStatus_t error, const char* file, int line)
{
  if (error != CURAND_STATUS_SUCCESS) {
    randutil_log().fatal() << "Internal CURAND failure with error " << (int)error << " in file "
                           << file << " at line " << line;
    assert(false);
  }
}

struct CPUGenerator : public CURANDGenerator {
  CPUGenerator(BitGeneratorType gentype, uint64_t seed, uint64_t generatorId, uint32_t flags)
    : CURANDGenerator(gentype, seed, generatorId)
  {
    CHECK_CURAND(::randutilCreateGeneratorHost(&gen_, type_, seed, generatorId, flags));
  }

  virtual ~CPUGenerator() { CHECK_CURAND(::randutilDestroyGenerator(gen_)); }
};

template <>
struct CURANDGeneratorBuilder<VariantKind::CPU> {
  static CURANDGenerator* build(BitGeneratorType gentype,
                                uint64_t seed,
                                uint64_t generatorId,
                                uint32_t flags)
  {
    return new CPUGenerator(gentype, seed, generatorId, flags);
  }

  static void destroy(CURANDGenerator* cugenptr) { delete cugenptr; }
};

template <>
std::map<legate::Processor, std::unique_ptr<generator_map<VariantKind::CPU>>>
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
