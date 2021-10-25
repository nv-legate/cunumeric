/* Copyright 2021 NVIDIA Corporation
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

#include "numpy/binary/binary_op_util.h"
#include "numpy/pitches.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <VariantKind KIND, BinaryOpCode OP_CODE, LegateTypeCode CODE, int DIM>
struct BinaryOpImplBody;

template <VariantKind KIND, BinaryOpCode OP_CODE>
struct BinaryOpImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<BinaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(BinaryOpArgs& args) const
  {
    using OP  = BinaryOp<OP_CODE, CODE>;
    using ARG = legate_type_of<CODE>;
    using RES = std::result_of_t<OP(ARG, ARG)>;

    auto rect = args.out.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;
    std::cout<<"rect1 "<<rect<<std::endl;
    std::cout<<"shape1 "<< args.in1.shape<DIM>()<<std::endl;
    std::cout<<"shape2 "<< args.in2.shape<DIM>()<<std::endl;


    auto out = args.out.write_accessor<RES, DIM>(rect);
    auto in1 = args.in1.read_accessor<ARG, DIM>(rect);
    auto in2 = args.in2.read_accessor<ARG, DIM>(rect);

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in1.accessor.is_dense_row_major(rect) &&
                 in2.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    OP func{args.args};
    BinaryOpImplBody<KIND, OP_CODE, CODE, DIM>()(func, out, in1, in2, pitches, rect, dense);
  }

  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<!BinaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(BinaryOpArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct BinaryOpDispatch {
  template <BinaryOpCode OP_CODE>
  void operator()(BinaryOpArgs& args) const
  {
    auto dim = std::max(1, args.out.dim());
    double_dispatch(dim, args.in1.code(), BinaryOpImpl<KIND, OP_CODE>{}, args);
  }
};

template <VariantKind KIND>
static void binary_op_template(TaskContext& context)
{
  /*
  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();
  auto& scalars = context.scalars();

  std::vector<Store> extra_args;
  for (size_t idx = 2; idx < inputs.size(); ++idx) extra_args.push_back(std::move(inputs[idx]));

  BinaryOpArgs args{
    inputs[0], inputs[1], outputs[0], scalars[0].value<BinaryOpCode>(), std::move(extra_args)};
  op_dispatch(args.op_code, BinaryOpDispatch<KIND>{}, args);
  */
  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();
  auto& scalars = context.scalars();
/*
  std::vector<UntypedScalar> extra_args;
  for (size_t idx = 2; idx < inputs.size(); ++idx)
    extra_args.push_back(inputs[idx].scalar<UntypedScalar>());
   std::cout<<"extra args size"<<extra_args.size()<<std::endl;
  BinaryOpArgs args{
    inputs[0], inputs[1], outputs[0], scalars[0].value<BinaryOpCode>(), std::move(extra_args)};
  op_dispatch(args.op_code, BinaryOpDispatch<KIND>{}, args);
*/
  //std::cout<<"fused inputs"<<inputs.size()<<std::endl;
  int nOps = scalars.size();// inputs[0].shape<1>().hi.x;
  //std::cout<<"nops "<<nOps<<std::endl;

  for (int i=0; i<scalars.size(); i++)
  {
    auto opcode = scalars[i].value<BinaryOpCode>();
    auto opcode_i = static_cast<std::underlying_type<BinaryOpCode>::type>(opcode);
    //std::cout<<"opcode "<<opcode_i<<std::endl;
  }
  for (int i=0; i<4; i++)
  {
    //std::cout<<inputs[i].shape<1>().lo.x<<" "<<inputs[i].shape<1>().hi.x<<std::endl;
  }
  std::vector<Store> extra_args;
  //for (size_t idx = 3; idx < inputs.size(); ++idx)
  //  extra_args.push_back(inputs[idx].scalar<UntypedScalar>());

  int inputStart=0;
  int outputStart=0;
 
  for (int i=0; i<nOps; i++)
  {
    std::cout<<"domains"<<std::endl;
    std::cout<<inputs[inputStart+0].domain()<<std::endl;
    std::cout<<inputs[inputStart+1].domain()<<std::endl;
    BinaryOpArgs args{
      inputs[inputStart+0], inputs[inputStart+1], outputs[outputStart],  scalars[outputStart].value<BinaryOpCode>(), std::move(extra_args)};
    std::cout<<"dispatching  "<<i<<std::endl;
    op_dispatch(args.op_code, BinaryOpDispatch<KIND>{}, args);
    inputStart+=2;
    outputStart+=1;
  }


}

}  // namespace numpy
}  // namespace legate
