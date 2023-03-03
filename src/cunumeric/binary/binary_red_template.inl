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
#include "cunumeric/binary/binary_red.h"
#include "cunumeric/binary/binary_op_util.h"
#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, BinaryOpCode OP_CODE, LegateTypeCode CODE, int DIM>
struct BinaryRedImplBody;

template <VariantKind KIND, BinaryOpCode OP_CODE>
struct BinaryRedImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<BinaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(BinaryRedArgs& args) const
  {
    using OP  = BinaryOp<OP_CODE, CODE>;
    using ARG = legate_type_of<CODE>;

    // A technical note: unlike region-backed stores that are partitionable, future-backed stores
    // are not partitionable and replicated to all point tasks, including their metadata.
    // This can lead to an unalignment between the input stores when one of them is future-backed,
    // because it sees the metadata for the entire store, whereas the other one gets that for
    // a subset. That unalignment itself simply denotes that this task in fact has no computation
    // to perform, but to identiy such cases, we need to compute the intersection of the stores'
    // shapes and check if it's empty.
    auto rect = args.in1.shape<DIM>().intersection(args.in2.shape<DIM>());

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (0 == volume) return;

    auto out = args.out.reduce_accessor<ProdReduction<bool>, true, 1>();
    auto in1 = args.in1.read_accessor<ARG, DIM>(rect);
    auto in2 = args.in2.read_accessor<ARG, DIM>(rect);

#ifndef LEGATE_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = in1.accessor.is_dense_row_major(rect) && in2.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    OP func(args.args);
    BinaryRedImplBody<KIND, OP_CODE, CODE, DIM>()(func, out, in1, in2, pitches, rect, dense);
  }

  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<!BinaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(BinaryRedArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct BinaryRedDispatch {
  template <BinaryOpCode OP_CODE>
  void operator()(BinaryRedArgs& args) const
  {
    auto dim = std::max(1, std::max(args.in1.dim(), args.in2.dim()));
    double_dispatch(dim, args.in1.code(), BinaryRedImpl<KIND, OP_CODE>{}, args);
  }
};

template <VariantKind KIND>
static void binary_red_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& scalars = context.scalars();

  std::vector<Store> extra_args;
  for (size_t idx = 2; idx < inputs.size(); ++idx) extra_args.push_back(std::move(inputs[idx]));

  BinaryRedArgs args{context.reductions()[0],
                     inputs[0],
                     inputs[1],
                     scalars[0].value<BinaryOpCode>(),
                     std::move(extra_args)};
  reduce_op_dispatch(args.op_code, BinaryRedDispatch<KIND>{}, args);
}

}  // namespace cunumeric
