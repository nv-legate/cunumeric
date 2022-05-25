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

#include "cunumeric/arg.h"
#include "cunumeric/pitches.h"

#include "bitgenerator_util.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND>
struct BitGeneratorImplBody;

template <VariantKind KIND>
struct BitGeneratorImpl {
  void operator()(BitGeneratorArgs& args) const
  {
    BitGeneratorImplBody<KIND>{}(args.bitgen_op,
                                 args.generatorID,
                                 args.generatorType,
                                 args.seed,
                                 args.flags,
                                 args.distribution,
                                 args.strides,
                                 args.intparams,
                                 args.floatparams,
                                 args.doubleparams,
                                 args.output,
                                 args.args);
  }
};

template <VariantKind KIND>
static void bitgenerator_template(TaskContext& context)
{
  auto& inputs       = context.inputs();
  auto& outputs      = context.outputs();
  auto& scalars      = context.scalars();
  auto bitgen_op     = scalars[0].value<BitGeneratorOperation>();
  auto generatorID   = scalars[1].value<int32_t>();
  auto generatorType = scalars[2].value<uint32_t>();
  auto seed          = scalars[3].value<uint64_t>();
  auto flags         = scalars[4].value<uint32_t>();

  BitGeneratorDistribution distribution;
  std::vector<int64_t> intparams;
  std::vector<float> floatparams;
  std::vector<double> doubleparams;

  DomainPoint strides;  // optional parameter
  legate::Span<const int32_t> todestroy;
  switch (bitgen_op) {
    case BitGeneratorOperation::DESTROY:  // gather same parameters as CREATE
    case BitGeneratorOperation::CREATE: {
      if (scalars.size() > 5) todestroy = scalars[5].values<int32_t>();
      break;
    }
    case BitGeneratorOperation::RAND_RAW: {
      if (scalars.size() > 5) strides = scalars[5].value<DomainPoint>();
      break;
    }
    case BitGeneratorOperation::DISTRIBUTION: {
      if (scalars.size() < 9) {
        ::fprintf(stderr, "FATAL: not enough parameters\n");
        ::exit(1);
      }
      distribution    = scalars[5].value<BitGeneratorDistribution>();
      auto _intparams = scalars[6].values<int64_t>();
      for (int k = 0; k < _intparams.size(); ++k) intparams.push_back(_intparams[k]);
      auto _floatparams = scalars[7].values<float>();
      for (int k = 0; k < _floatparams.size(); ++k) floatparams.push_back(_floatparams[k]);
      auto _doubleparams = scalars[8].values<double>();
      for (int k = 0; k < _doubleparams.size(); ++k) doubleparams.push_back(_doubleparams[k]);
      break;
    }
  }

  std::vector<Store> extra_args;
  for (auto& input : inputs) extra_args.push_back(std::move(input));

  std::vector<Store> optional_output;
  for (auto& output : outputs) optional_output.push_back(std::move(output));

  // destroy ?
  for (int k = 0; k < todestroy.size(); ++k) {
    BitGeneratorArgs dargs = BitGeneratorArgs::destroy(todestroy[k]);
    BitGeneratorImpl<KIND>{}(dargs);
  }

  BitGeneratorArgs args(bitgen_op,
                        generatorID,
                        generatorType,
                        seed,
                        flags,
                        distribution,
                        strides,
                        std::move(intparams),
                        std::move(floatparams),
                        std::move(doubleparams),
                        std::move(optional_output),
                        std::move(extra_args));
  BitGeneratorImpl<KIND>{}(args);
}

}  // namespace cunumeric