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

#pragma once

#include "cunumeric/cunumeric.h"
#include "cunumeric/random/bitgenerator_util.h"

namespace cunumeric {

struct BitGeneratorArgs {
  BitGeneratorOperation bitgen_op;
  int32_t generatorID;
  BitGeneratorType generatorType;
  uint64_t seed;
  uint32_t flags;

  BitGeneratorDistribution distribution;

  legate::DomainPoint strides;
  std::vector<int64_t> intparams;
  std::vector<float> floatparams;
  std::vector<double> doubleparams;

  std::vector<legate::Store> output;  // size 0 or 1
  std::vector<legate::Store> args;

  BitGeneratorArgs() {}
  static BitGeneratorArgs destroy(int32_t id)
  {
    BitGeneratorArgs res;
    res.bitgen_op   = BitGeneratorOperation::DESTROY;
    res.generatorID = id;
    return res;
  }

  BitGeneratorArgs(BitGeneratorOperation bitgen_op,
                   int32_t generatorID,
                   BitGeneratorType generatorType,
                   uint64_t seed,
                   uint32_t flags,

                   BitGeneratorDistribution distribution,

                   legate::DomainPoint strides,
                   std::vector<int64_t>&& intparams,
                   std::vector<float>&& floatparams,
                   std::vector<double>&& doubleparams,

                   std::vector<legate::Store>&& output,  // size 0 or 1
                   std::vector<legate::Store>&& args)
    : bitgen_op(bitgen_op),
      generatorID(generatorID),
      generatorType(generatorType),
      seed(seed),
      flags(flags),
      distribution(distribution),
      strides(strides),
      intparams(std::move(intparams)),
      floatparams(std::move(floatparams)),
      doubleparams(std::move(doubleparams)),
      output(std::move(output)),
      args(std::move(args))
  {
  }
};

class BitGeneratorTask : public CuNumericTask<BitGeneratorTask> {
 public:
  static const int TASK_ID = CUNUMERIC_BITGENERATOR;

 public:
  static void cpu_variant(legate::TaskContext& context);
#ifdef LEGATE_USE_OPENMP
  // TODO: Fully parallelized OpenMP implementation for BitGenerator
  // Doing it this way is safe, but only one thread is being used out of the OpenMP pool.
  static void omp_variant(legate::TaskContext& context) { BitGeneratorTask::cpu_variant(context); }
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

}  // namespace cunumeric
