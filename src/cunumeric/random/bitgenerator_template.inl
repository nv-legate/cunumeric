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
    BitGeneratorImplBody<KIND>{}(args.bitgen_op, args.generatorID, args.parameter, args.output, args.args);
  }
};

template <VariantKind KIND>
static void bitgenerator_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();
  auto& scalars = context.scalars();
  auto bitgen_op   = scalars[0].value<BitGeneratorOperation>();
  auto generatorID = scalars[1].value<int32_t>();
  auto parameter   = scalars[2].value<uint64_t>();

  std::vector<Store> extra_args;
  for (auto& input : inputs) extra_args.push_back(std::move(input));

  std::vector<Store> optional_output;
  for (auto& output : outputs) optional_output.push_back(std::move(output));

  BitGeneratorArgs args{bitgen_op, generatorID, parameter, std::move(optional_output), std::move(extra_args)};
  BitGeneratorImpl<KIND>{}(args);
}

}  // namespace cunumeric