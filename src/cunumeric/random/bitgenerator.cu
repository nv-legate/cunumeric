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
#include "cunumeric/random/bitgenerator_util.h"

#include "cunumeric/cuda_help.h"
#include "cunumeric/random/curand_help.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

struct CURANDGenerator
{
  static constexpr size_t DEFAULT_DEV_BUFFER_SIZE = 64*1024 ; // TODO: optimize this
  curandGenerator_t gen ;
  uint64_t seed ;
  uint64_t offset ;
  curandRngType type ;
  bool supports_skipahead ; 
  size_t dev_buffer_size ; // in number of entries
  uint32_t* dev_buffer ; // buffer for intermediate results
};

template<>
struct BitGeneratorImplBody<VariantKind::GPU> {
  thread_local static std::map<int, CURANDGenerator> m_generators ;

  void operator()(BitGeneratorOperation op,
        int32_t generatorID, 
        uint64_t parameter,
        std::vector<legate::Store>& output,
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
          CURANDGenerator cugen ;
          cugen.gen = gen ;
          cugen.offset = 0 ;
          cugen.type = get_curandRngType((BitGeneratorType)parameter) ;
          cugen.supports_skipahead = supportsSkipAhead(cugen.type) ;
          cugen.dev_buffer_size = cugen.DEFAULT_DEV_BUFFER_SIZE ;
          CHECK_CUDA(::cudaMalloc(&(cugen.dev_buffer), cugen.dev_buffer_size * sizeof(uint32_t)));
          m_generators[generatorID] = cugen ;
        }
        break ;
      case BitGeneratorOperation::DESTROY:
        {
          if (m_generators.find(generatorID) == m_generators.end())
          {
            ::fprintf(stderr, "[ERROR] : internal error : generator ID <%d> does not exist !\n", generatorID);
            ::exit(1); 
          }
          CURANDGenerator& cugen = m_generators[generatorID] ;
          CHECK_CUDA(::cudaFree(cugen.dev_buffer));
          CHECK_CURAND(::curandDestroyGenerator(cugen.gen));
          m_generators.erase(generatorID);
        }
        break ;
      case BitGeneratorOperation::RAND_RAW:
        {
          if (m_generators.find(generatorID) == m_generators.end())
          {
            ::fprintf(stderr, "[ERROR] : internal error : generator ID <%d> does not exist !\n", generatorID);
            ::exit(1); 
          }

          if (output.size() == 0)
          {
            CURANDGenerator& cugen = m_generators[generatorID] ;
            if (cugen.supports_skipahead)
            {
              // skip ahead
              ::fprintf(stdout, "[DEBUG] : @ %s : %d -- parameter = %lu - offset = %lu\n", __FILE__, __LINE__, parameter, cugen.offset);
              cugen.offset += parameter ;
              CHECK_CURAND(::curandSetGeneratorOffset(cugen.gen, cugen.offset));
            } else {
              // actually generate numbers in the temporary buffer
              ::fprintf(stdout, "[DEBUG] : @ %s : %d -- parameter = %lu - offset = %lu\n", __FILE__, __LINE__, parameter, cugen.offset);
              uint64_t remain = parameter ;
              while (remain > 0)
              {
                if (remain < cugen.dev_buffer_size)
                {
                  CHECK_CURAND(::curandGenerate(cugen.gen, cugen.dev_buffer, (size_t)remain));
                  break ;
                }
                else
                  CHECK_CURAND(::curandGenerate(cugen.gen, cugen.dev_buffer, (size_t)cugen.dev_buffer_size));
                remain -= cugen.dev_buffer_size ;
              }
            }
          } else {
            ::fprintf(stderr, "[ERROR] : @ %s : %d -- not implemented !\n", __FILE__, __LINE__);
            ::exit(1);
          }
        }
        break;
      default:
        {
          ::fprintf(stderr, "[ERROR] : unknown BitGenerator operation") ;
          ::exit(1);              
        }
    }
  }
};

thread_local std::map<int,CURANDGenerator> BitGeneratorImplBody<VariantKind::GPU>::m_generators ;

/*static*/ void BitGeneratorTask::gpu_variant(legate::TaskContext& context)
{
    bitgenerator_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric