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

#include <map>

#include "cunumeric/random/bitgenerator.h"
#include "cunumeric/random/bitgenerator_template.inl"

#include "cunumeric/random/curand_help.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

static inline curandRngType get_curandRngType(BitGeneratorType kind)
{
  switch (kind)
  {
    case BitGeneratorType::DEFAULT:
      return curandRngType::CURAND_RNG_PSEUDO_XORWOW ;
    case BitGeneratorType::XORWOW:
      return curandRngType::CURAND_RNG_PSEUDO_XORWOW ;
    case BitGeneratorType::MRG32K3A:
      return curandRngType::CURAND_RNG_PSEUDO_MRG32K3A ;
    case BitGeneratorType::MTGP32:
      return curandRngType::CURAND_RNG_PSEUDO_MTGP32 ;
    case BitGeneratorType::MT19937:
      return curandRngType::CURAND_RNG_PSEUDO_MT19937 ;
    case BitGeneratorType::PHILOX4_32_10:
      return curandRngType::CURAND_RNG_PSEUDO_PHILOX4_32_10 ;
    default:
      {
        ::fprintf(stderr, "[ERROR] : unknown generator") ;
        ::exit(1);
      }
  }
}

template<>
struct BitGeneratorImplBody<VariantKind::GPU> {
  thread_local static std::map<int, curandGenerator_t> m_generators ;
  void operator()(BitGeneratorOperation op,
        int32_t generatorID, 
        uint64_t parameter,
        std::vector<legate::Store>& args)
  {
      switch (op)
      {
        case BitGeneratorOperation::CREATE:
          {
            if (m_generators.find(generatorID) != m_generators.end())
            {
              ::fprintf(stderr, "[ERROR] : internal error : generator ID <%d> already in use !\n", generatorID);
              ::exit(1); 
            }
            curandGenerator_t gen ;
            CHECK_CURAND(::curandCreateGenerator(&gen, get_curandRngType((BitGeneratorType)parameter)));
            m_generators[generatorID] = gen ;
          }
          break ;
        case BitGeneratorOperation::DESTROY:
          {
            if (m_generators.find(generatorID) == m_generators.end())
            {
              ::fprintf(stderr, "[ERROR] : internal error : generator ID <%d> already in use !\n", generatorID);
              ::exit(1); 
            }
            curandGenerator_t gen = m_generators[generatorID];
            CHECK_CURAND(::curandDestroyGenerator(gen));
            m_generators.erase(generatorID);
          }
          break ;
        default:
          {
            ::fprintf(stderr, "[ERROR] : unknown BitGenerator operation") ;
            ::exit(1);              
          }
      }
  }
};

thread_local std::map<int,curandGenerator_t> BitGeneratorImplBody<VariantKind::GPU>::m_generators ;

/*static*/ void BitGeneratorTask::gpu_variant(legate::TaskContext& context)
{
    printf("[INFO] : @ %s : %d\n", __FILE__, __LINE__);
    bitgenerator_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric