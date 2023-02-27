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

// Useful for IDEs
#include "cunumeric/random/rand.h"
#include "cunumeric/arg.h"
#include "cunumeric/arg.inl"
#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, typename RNG, typename VAL, int DIM>
struct RandImplBody;

template <RandGenCode GEN_CODE, VariantKind KIND>
struct RandImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<RandomGenerator<GEN_CODE, CODE>::valid>* = nullptr>
  void operator()(RandArgs& args) const
  {
    using VAL = legate_type_of<CODE>;
    using RNG = RandomGenerator<GEN_CODE, CODE>;

    auto rect = args.out.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out = args.out.write_accessor<VAL, DIM>(rect);
    Point<DIM> strides(args.strides);

    RNG rng(args.epoch, args.args);
    RandImplBody<KIND, RNG, VAL, DIM>{}(out, rng, strides, pitches, rect);
  }

  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<!RandomGenerator<GEN_CODE, CODE>::valid>* = nullptr>
  void operator()(RandArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct RandDispatch {
  template <RandGenCode GEN_CODE>
  void operator()(RandArgs& args) const
  {
    double_dispatch(args.out.dim(), args.out.code(), RandImpl<GEN_CODE, KIND>{}, args);
  }
};

template <VariantKind KIND>
static void rand_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();
  auto& scalars = context.scalars();

  auto gen_code = scalars[0].value<RandGenCode>();
  auto epoch    = scalars[1].value<uint32_t>();
  auto strides  = scalars[2].value<DomainPoint>();

  std::vector<Store> extra_args;
  for (auto& input : inputs) extra_args.push_back(std::move(input));

  RandArgs args{outputs[0], gen_code, epoch, strides, std::move(extra_args)};
  op_dispatch(args.gen_code, RandDispatch<KIND>{}, args);
}

}  // namespace cunumeric
