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

#include "cunumeric/fused/fused_op.h"
//#include "cunumeric/binary/binary_op_template.inl"

#include "cunumeric/cuda_help.h"

//namespace legate {
namespace cunumeric {

using namespace Legion;
using namespace legate;
/*
template <typename Function, typename RES, typename ARG>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  dense_kernel(size_t volume, Function func, RES* out, const ARG* in1, const ARG* in2)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  out[idx] = func(in1[idx], in2[idx]);
}

template <typename Function, typename WriteAcc, typename ReadAcc, typename Pitches, typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) generic_kernel(
  size_t volume, Function func, WriteAcc out, ReadAcc in1, ReadAcc in2, Pitches pitches, Rect rect)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto point = pitches.unflatten(idx, rect.lo);
  out[point] = func(in1[point], in2[point]);
}

template <BinaryOpCode OP_CODE, LegateTypeCode CODE, int DIM>
struct BinaryOpImplBody<VariantKind::GPU, OP_CODE, CODE, DIM> {
  using OP  = BinaryOp<OP_CODE, CODE>;
  using ARG = legate_type_of<CODE>;
  using RES = std::result_of_t<OP(ARG, ARG)>;

  void operator()(OP func,
                  AccessorWO<RES, DIM> out,
                  AccessorRO<ARG, DIM> in1,
                  AccessorRO<ARG, DIM> in2,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    size_t volume       = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dense) {
      auto outptr = out.ptr(rect);
      auto in1ptr = in1.ptr(rect);
      auto in2ptr = in2.ptr(rect);
      dense_kernel<<<blocks, THREADS_PER_BLOCK>>>(volume, func, outptr, in1ptr, in2ptr);
    } else {
      generic_kernel<<<blocks, THREADS_PER_BLOCK>>>(volume, func, out, in1, in2, pitches, rect);
    }
  }
};
*/
/*static*/ void FusedOpTask::gpu_variant(TaskContext& context){

  int nOps = context.fusionMetadata.nOps;
  auto opIDs = context.fusionMetadata.opIDs;
  auto offsets = context.fusionMetadata.offsets;
  for (int i=0; i<nOps; i++)
  {
      //packOp(ms, context, i);
      std::vector<Legion::PhysicalRegion> regions;
      //create new context
      const Legion::Task* task = (Legion::Task*) context.task_;

      //pack inputs
      std::vector<Store> inputs;
      auto inputStarts = context.fusionMetadata.inputStarts;
      auto outputStarts = context.fusionMetadata.outputStarts;
      auto offsetStarts = context.fusionMetadata.offsetStarts;
      auto reductionStarts = context.fusionMetadata.reductionStarts;
      unsigned nInputs = (inputStarts[i+1]-inputStarts[i]); //want to pack this as a 32 bit uint 
      for (unsigned j = 0; j<nInputs; j++)
      {
          int offsetStart = offsetStarts[i];
          int inputStart = inputStarts[i];
          int bufferID = offsets[offsetStart+j]-1;
          Store& input = context.inputs()[inputStart+bufferID];
          inputs.push_back(std::move(input));
      }

      //pack outputs
      std::vector<Store> outputs;
      unsigned nOutputs = (outputStarts[i+1]-outputStarts[i]); //want to pack this as a 32 bit uint 
      for (unsigned j = 0; j<nOutputs; j++)
      {
          int offsetStart = offsetStarts[i];
          int outputStart = outputStarts[i];
          int bufferID = offsets[offsetStart+nInputs+j];
          bufferID = (-bufferID)-1;
              Store& output = context.outputs()[outputStart+bufferID];
              outputs.push_back(std::move(output));
      }

      //pack reductions
      std::vector<Store> reductions;
      int32_t nReductions = (reductionStarts[i+1]-reductionStarts[i]);
      for (unsigned j = 0; j<nReductions; j++)
      {
          int offsetStart = offsetStarts[i];
          int reductionStart = reductionStarts[i];
          int bufferID = offsets[offsetStart+nInputs+nOutputs+j];
          //all buffer ids are 1 -indexed
          //negative id is an output, while a positive id is an output
          if (bufferID<0) 
          {
              bufferID = (-bufferID)-1;
              Store& reduction = context.reductions()[reductionStart+bufferID];
              reductions.push_back(std::move(reduction));
          }
      }

      //pack scalars
      //std::vector<Scalar> scalars;
      auto scalarStarts = context.fusionMetadata.scalarStarts;
      int32_t nScalars = (scalarStarts[i+1]-scalarStarts[i]);
      std::vector<Scalar> scalars;
      for (unsigned j = 0; j<nScalars; j++)
      {  
          scalars.push_back(std::move(context.scalars()[scalarStarts[i]+j]));
      }


      TaskContext context3(task, (const std::vector<Legion::PhysicalRegion>) regions);// inputs, outputs, scalars);
      context3.inputs_ = std::move(inputs);
      context3.outputs_ = std::move(outputs); 
      context3.reductions_ = std::move(reductions);
      context3.scalars_ = std::move(scalars);

      //launch
      auto descp = Core::gpuDescriptors.find(opIDs[i]);

      auto desc = descp->second;
      desc(context3);
      for (unsigned j = 0; j<nOutputs; j++)
      {
          int offsetStart = offsetStarts[i];
          int outputStart = outputStarts[i];
          int bufferID = offsets[offsetStart+nInputs+j];
          bufferID = (-bufferID)-1;
          context.outputs_[outputStart+bufferID] = std::move(context3.outputs_[j]);
      }
 

      //context3.pack_return_values();
      //context.pack_return_values();
  }
}

}  // namespace numpy
//}  // namespace legate
