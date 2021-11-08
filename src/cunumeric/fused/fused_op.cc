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

#include "core/utilities/makeshift_serializer.h"
#include "core/runtime/runtime.h"
#include "core/runtime/context.h"
#include "cunumeric/fused/binary_op.h"
#include "cunumeric/fused/binary_op_template.inl"
#include "legion.h"
#include <set>
#include <time.h>
#include <sys/time.h>

namespace legate {
namespace cunmeric {

using namespace Legion;

template <BinaryOpCode OP_CODE, LegateTypeCode CODE, int DIM>
struct BinaryOpImplBody<VariantKind::CPU, OP_CODE, CODE, DIM> {
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
    const size_t volume = rect.volume();
    if (dense) {
      auto outptr = out.ptr(rect);
      auto in1ptr = in1.ptr(rect);
      auto in2ptr = in2.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) outptr[idx] = func(in1ptr[idx], in2ptr[idx]);
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        out[p] = func(in1[p], in2[p]);
      }
    }
  }
};



//op id refers not to the op's type, but the index in the list of fused ops
void packOp(MakeshiftSerializer& ms, TaskContext& context, int opID)
{
  auto inputStarts = context.fusionMetadata.inputStarts;
  auto outputStarts = context.fusionMetadata.outputStarts;
  auto offsetStarts = context.fusionMetadata.offsetStarts;
  auto offsets =  context.fusionMetadata.offsets;
  auto reductionStarts =  context.fusionMetadata.reductionStarts;
  auto scalarStarts = context.fusionMetadata.scalarStarts;

  //the leaf task cannot be fused ops currently
  ms.pack((bool) false);

  //pack inputs
  unsigned nInputs = (inputStarts[opID+1]-inputStarts[opID]); //want to pack this as a 32 bit uint 
  ms.pack((uint32_t) nInputs);
  //std::cout<<"inputs "<<nInputs<<std::endl;
  for (unsigned i = 0; i<nInputs; i++)
  {
      int offsetStart = offsetStarts[opID];
      int inputStart = inputStarts[opID];
      int bufferID = offsets[offsetStart+i];
      //all buffer id are 1 -indexed
      //negative id is an output, while a positive id is an input
      if (bufferID>0) 
      {
          bufferID--;
          //std::cout<<"packing input "<<bufferID<<std::endl;
          //Store& input = context.inputs()[nMetaDataArrs+inputStart+bufferID];
          Store& input = context.inputs()[inputStart+bufferID];
          ms.addReqID(input.getReqIdx());
          ms.packBuffer(input);
      }
  }

  unsigned nOutputs = (outputStarts[opID+1]-outputStarts[opID]); //want to pack this as a 32 bit uint 
  ms.pack((uint32_t) nOutputs);
  //std::cout<<"outputs "<<nOutputs<<std::endl;
  //pack outputs
  for (unsigned i = 0; i<nOutputs; i++)
  {
      int offsetStart = offsetStarts[opID];
      int outputStart = outputStarts[opID];
      int bufferID = offsets[offsetStart+nInputs+i];
      //all buffer ids are 1 -indexed
      //negative id is an output, while a positive id is an output
      if (bufferID<0) 
      {
          bufferID = (-bufferID)-1;
          Store& output = context.outputs()[outputStart+bufferID];
          ms.addReqID(output.getReqIdx());
          ms.packBuffer(output);
      }
  }

  //pack reductions
  int32_t nReductions = (reductionStarts[opID+1]-reductionStarts[opID]);
  ms.pack((uint32_t) nReductions);

  //pack scalars
  int32_t nScalars = (scalarStarts[opID+1]-scalarStarts[opID]);
  ms.pack((uint32_t) nScalars);
  for (unsigned i = 0; i<nScalars; i++)
  {  
      ms.packScalar(context.scalars()[scalarStarts[opID]+i]);
  }
}

/*static*/ void FusedOpTask::cpu_variant(TaskContext& context)
{
  int nOps = context.fusionMetadata.nOps;
  MakeshiftSerializer ms;
  auto opIDs = context.fusionMetadata.opIDs;
  auto offsets = context.fusionMetadata.offsets;
  for (int i=0; i<nOps; i++)
  {
      ms.zero(); //reset the serializer, but keep the memory
      packOp(ms, context, i);
     /*
      TaskLauncher leaf_launcher(opIDs[i], TaskArgument(ms.ptr(), ms.buffSize())); 
      leaf_launcher.enable_inlining=true;
      leaf_launcher.point = context.task_->index_point;
      //add appropriate region requirements for this op
      std::vector<int32_t> reqIds = ms.getReqIds();
      for (int32_t reqIdx : reqIds)
      {
          auto& req = context.task_->regions[reqIdx];
          leaf_launcher.add_region_requirement(req);
      } 
      //add appropriate futures
      auto futureStarts = context.fusionMetadata.futureStarts;
      for (int32_t fuid = futureStarts[i]; fuid<futureStarts[i+1]; fuid++)
      {
          auto& f = context.task_->futures[fuid];
          leaf_launcher.add_future(f);
      } 

      */
      std::vector<Legion::PhysicalRegion> regions;
      //for (int k : reqIds)
      {
          //regions.push_back(context.regions_[k]);
      }     

      //context.runtime_->execute_task(context.context_, leaf_launcher);

      //create new context
      const Legion::Task* task = (Legion::Task*) context.task_;

      //pack inputs
      std::vector<Store> inputs;
      auto inputStarts = context.fusionMetadata.inputStarts;
      auto outputStarts = context.fusionMetadata.outputStarts;
      auto offsetStarts = context.fusionMetadata.offsetStarts;
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

      //pack scalars
      //std::vector<Scalar> scalars;
      auto scalarStarts = context.fusionMetadata.scalarStarts;
      int32_t nScalars = (scalarStarts[i+1]-scalarStarts[i]);
      for (unsigned j = 0; j<nScalars; j++)
      {  
        //  scalars.push_back(std::move(context.scalars()[scalarStarts[i]+j]));
      }

      /*
      std::vector<Store> outputs;
      outputs.push_back(std::move(context.outputs()[i+0]));
      */
      std::vector<Scalar> scalars;
      scalars.push_back(std::move(context.scalars()[i+0]));

      TaskContext context3(task, (const std::vector<Legion::PhysicalRegion>) regions);// inputs, outputs, scalars);
      context3.inputs_ = std::move(inputs);
      context3.outputs_ = std::move(outputs); 
      context3.scalars_ = std::move(scalars);

      //launch
      auto descp = Core::cpuDescriptors.find(opIDs[i]);
      auto desc = descp->second;
      desc(context3);

      //move it back??
      //context.outputs_[i] = std::move(context3.outputs_[i]);
     
      //binary_op_template<VariantKind::CPU>(context);
    //gettimeofday(&start, NULL);
      //gettimeofday(&end, NULL);
 // printf("%ld\n", ((end.tv_sec * 1000000 + end.tv_usec)
//		  - (start.tv_sec * 1000000 + start.tv_usec)));
  }
  

}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { FusedOpTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
