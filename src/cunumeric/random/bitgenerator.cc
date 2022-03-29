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

#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <>
struct BitGeneratorImplBody<VariantKind::CPU> {
  thread_local static std::map<int, curandGenerator_t> m_generators;

  void operator()(BitGeneratorOperation op,
                  int32_t generatorID,
                  uint64_t parameter,
                  const DomainPoint& strides,
                  std::vector<legate::Store>& output,
                  std::vector<legate::Store>& args)
  {
    // ::fprintf(stderr, "[TRACE] : bitgenerator tid = %d\n", syscall(SYS_gettid));
    switch (op) {
      case BitGeneratorOperation::CREATE: {
        if (m_generators.find(generatorID) != m_generators.end()) {
          ::fprintf(
            stderr, "[ERROR] : internal error : generator ID <%d> already in use !\n", generatorID);
          assert(false);
        }
        curandGenerator_t gen;
        CHECK_CURAND(
          ::curandCreateGeneratorHost(&gen, get_curandRngType((BitGeneratorType)parameter)));
        m_generators[generatorID] = gen;
      } break;
      case BitGeneratorOperation::DESTROY: {
        if (m_generators.find(generatorID) == m_generators.end()) {
          ::fprintf(
            stderr, "[ERROR] : internal error : generator ID <%d> does not exist !\n", generatorID);
          assert(false);
        }
        curandGenerator_t gen = m_generators[generatorID];
        CHECK_CURAND(::curandDestroyGenerator(gen));
        m_generators.erase(generatorID);
      } break;
      default: {
        ::fprintf(stderr, "[ERROR] : unknown BitGenerator operation");
        assert(false);
      }
    }
  }
};

thread_local std::map<int, curandGenerator_t> BitGeneratorImplBody<VariantKind::CPU>::m_generators;

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