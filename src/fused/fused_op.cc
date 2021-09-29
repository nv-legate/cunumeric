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

#include "utilities/makeshift_serializer.h"
#include "fused/binary_op.h"
#include "fused/binary_op_template.inl"
#include "legion.h"


namespace legate {
namespace numpy {

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

void inline_leaf_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
  printf("Hello from 'inline_leaf_task' being inlined into leaf 'fused op'\n");
}

//op id refers not to the op's type, but the index in the list of fused ops
void packOp(MakeshiftSerializer& ms, TaskContext& context, int opID)
{
  //grab all the stores
  const int nMetaDataArrs=7;
  Store& inputStartsStore = context.inputs()[0];
  Store& outputStartsStore = context.inputs()[1];
  Store& offsetStartsStore = context.inputs()[2];
  Store& offsetsStore = context.inputs()[3];
  Store& reductionStartsStore = context.inputs()[4];
  Store& scalarStartsStore = context.inputs()[5];

  //grab pointers to all the metadata arrays
  using ARG = legate_type_of<LegateTypeCode::INT64_LT>;
  auto opRect = context.inputs()[0].shape<1>(); //iterator over range(0,nOps)
  auto offsetMetaRect = context.inputs()[3].shape<1>(); //iter over range(0, nInputs+nOutputs)

  auto inputStarts = inputStartsStore.read_accessor<ARG, 1>().ptr(opRect);
  auto outputStarts = outputStartsStore.read_accessor<ARG, 1>().ptr(opRect);
  auto offsetStarts = offsetStartsStore.read_accessor<ARG, 1>().ptr(opRect);
  auto offsets = offsetsStore.read_accessor<ARG, 1>().ptr(offsetMetaRect);
  auto reductionStarts = reductionStartsStore.read_accessor<ARG, 1>().ptr(opRect);
  auto scalarStarts = scalarStartsStore.read_accessor<ARG, 1>().ptr(opRect);

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
          Store& input = context.inputs()[nMetaDataArrs+inputStart+bufferID];
          //const RegionRequirement& Physical::Regionget_requirement(void) const;
          //inline RegionRequirement& TaskLauncher::add_region_requirement(const RegionRequirement &req)
          //vim legion//runtime/legion.h line 3614 we can get all the region requirements
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
          //std::cout<<"packing output "<<bufferID<<std::endl;
          Store& output = context.outputs()[outputStart+bufferID];
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
  //const int INLINE_LEAF_TASK_ID =0;
  int nOps = context.inputs()[0].shape<1>().hi.x;
  auto offsetMetaRect = context.inputs()[3].shape<1>();
  //std::cout<<offsetMetaRect<<std::endl;
  using ARG = legate_type_of<LegateTypeCode::INT64_LT>;
  const Store& offsetsStore = context.inputs()[3];
  auto offsets = offsetsStore.read_accessor<ARG, 1>().ptr(offsetMetaRect);

  const Store& taskIDStore = context.inputs()[6];
  auto opRect = context.inputs()[6].shape<1>();
  auto taskIDs = offsetsStore.read_accessor<ARG, 1>().ptr(opRect);

  //pack inputs
  for (int i=0; i<nOps; i++)
  {
      MakeshiftSerializer ms;
      packOp(ms, context, i);
      TaskLauncher leaf_launcher(1074141829, TaskArgument(ms.ptr(), ms.buffSize()+8)); 
      leaf_launcher.point = context.task_->index_point;
      for (int i=0; i< context.task_->regions.size();i++)
      {
          auto& req = context.task_->regions[i];
          leaf_launcher.add_region_requirement(req);
      }

      leaf_launcher.enable_inlining=true;
      context.runtime_->execute_task(context.context_, leaf_launcher);
  }
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { FusedOpTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
