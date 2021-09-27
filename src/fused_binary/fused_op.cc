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

#include "fused/makeshift_serializer.h"
#include "fused/binary_op.h"
#include "fused/binary_op_template.inl"
#include "legion.h"
//#include "legion/legion_utilities.h"
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

/*static*/ void FusedOpTask::cpu_variant(TaskContext& context)
{
/*
TaskLauncher(TaskID tid, 
                   TaskArgument arg,
                   Predicate pred = Predicate::TRUE_PRED,
                   MapperID id = 0,
                   MappingTagID tag = 0);

  Deserializer dez(task, regions);
  inputs_     = dez.unpack<std::vector<Store>>();
  outputs_    = dez.unpack<std::vector<Store>>();
  reductions_ = dez.unpack<std::vector<Store>>();
  scalars_    = dez.unpack<std::vector<Scalar>>();
*/
  const int INLINE_LEAF_TASK_ID =0;
  MakeshiftSerializer ms;

  //pack inputs
  ms.pack(context.inputs().size());
  for (auto& input : context.inputs())
  {
      ms.pack(input.is_future()); //is_future
      ms.pack(input.dim());
      int code = input.code();
      ms.pack(code);
  }
  //pack inputs
  //pack outputs
  //pack reductions
  //pack scalars


  TaskLauncher inline_leaf_launcher(INLINE_LEAF_TASK_ID, TaskArgument()); 
  std::cout<<"trying to launch leaf task"<<std::endl;

  //Deserializer dez(task, regions);
  //Legion::Serializer rez;
  //Deserializer rez;
  //rez.serialize(context.inputs());
  //inline_leaf_launcher.enable_inlining = true;
  //runtime->execute_task(ctx, inline_leaf_launcher);


  //../legate.core/legion//examples/local_function_tasks/local_function_tasks.cc
  binary_op_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { FusedOpTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
