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

#include "numpy.h"
#include "mapper.h"
#include "proj.h"
#include "unary/unary_red_util.h"

using namespace Legion;

namespace legate {
namespace numpy {

static const char* const numpy_library_name = "legate.numpy";

#ifdef LEGATE_USE_CUDA
extern void register_gpu_reduction_operators(ReductionOpID first_redop_id);
#endif

/*static*/ void LegateNumPy::record_variant(TaskID tid,
                                            const char* task_name,
                                            const CodeDescriptor& descriptor,
                                            ExecutionConstraintSet& execution_constraints,
                                            TaskLayoutConstraintSet& layout_constraints,
                                            LegateVariant var,
                                            Processor::Kind kind,
                                            bool leaf,
                                            bool inner,
                                            bool idempotent,
                                            bool ret_type)
{
  assert((kind == Processor::LOC_PROC) || (kind == Processor::TOC_PROC) ||
         (kind == Processor::OMP_PROC));
  std::deque<PendingTaskVariant>& pending_task_variants = get_pending_task_variants();
  // Buffer these up until we can do our actual registration with the runtime
  pending_task_variants.push_back(PendingTaskVariant(
    tid,
    false /*global*/,
    (kind == Processor::LOC_PROC) ? "CPU" : (kind == Processor::TOC_PROC) ? "GPU" : "OpenMP",
    task_name,
    descriptor,
    var,
    ret_type));
  TaskVariantRegistrar& registrar = pending_task_variants.back();
  registrar.execution_constraints.swap(execution_constraints);
  registrar.layout_constraints.swap(layout_constraints);
  registrar.add_constraint(ProcessorConstraint(kind));
  registrar.set_leaf(leaf);
  registrar.set_inner(inner);
  registrar.set_idempotent(idempotent);
  // Everyone is doing registration on their own nodes
  registrar.global_registration = false;
}

/*static*/ std::deque<LegateNumPy::PendingTaskVariant>& LegateNumPy::get_pending_task_variants(void)
{
  static std::deque<PendingTaskVariant> pending_task_variants;
  return pending_task_variants;
}

void registration_callback(Machine machine,
                           Runtime* runtime,
                           const std::set<Processor>& local_procs)
{
  // This is the callback that we get from the runtime after it has started
  // but before the actual application starts running so we can now do all
  // our registrations.
  // First let's get our range of task IDs for this library from the runtime
  const size_t max_numpy_tasks = NUMPY_MAX_TASKS;
  const TaskID first_tid = runtime->generate_library_task_ids(numpy_library_name, max_numpy_tasks);
  std::deque<LegateNumPy::PendingTaskVariant>& pending_task_variants =
    LegateNumPy::get_pending_task_variants();
  // Do all our registrations
  for (std::deque<LegateNumPy::PendingTaskVariant>::iterator it = pending_task_variants.begin();
       it != pending_task_variants.end();
       it++) {
    // Make sure we haven't exceed our maximum range of IDs
    assert(it->task_id < max_numpy_tasks);
    it->task_id += first_tid;  // Add in our library offset
    // Attach the task name too for debugging
    runtime->attach_name(it->task_id, it->task_name, false /*mutable*/, true /*local only*/);
    runtime->register_task_variant(*it, it->descriptor, NULL, 0, it->ret_type, it->var);
  }
  pending_task_variants.clear();
  // Register our special reduction functions
  const ReductionOpID first_redop_id =
    runtime->generate_library_reduction_ids(numpy_library_name, NUMPY_MAX_REDOPS);
#ifdef LEGATE_USE_CUDA
  register_gpu_reduction_operators(first_redop_id);
#else
  REGISTER_ALL_REDUCTIONS(ArgmaxReduction, first_redop_id);
  REGISTER_ALL_REDUCTIONS(ArgminReduction, first_redop_id);
#endif

  Runtime::register_reduction_op<UntypedScalarRedOp<UnaryRedCode::MAX>>(first_redop_id +
                                                                        NUMPY_SCALAR_MAX_REDOP);
  Runtime::register_reduction_op<UntypedScalarRedOp<UnaryRedCode::MIN>>(first_redop_id +
                                                                        NUMPY_SCALAR_MIN_REDOP);
  Runtime::register_reduction_op<UntypedScalarRedOp<UnaryRedCode::PROD>>(first_redop_id +
                                                                         NUMPY_SCALAR_PROD_REDOP);
  Runtime::register_reduction_op<UntypedScalarRedOp<UnaryRedCode::SUM>>(first_redop_id +
                                                                        NUMPY_SCALAR_SUM_REDOP);
  Runtime::register_reduction_op<UntypedScalarRedOp<UnaryRedCode::ARGMAX>>(
    first_redop_id + NUMPY_SCALAR_ARGMAX_REDOP);
  Runtime::register_reduction_op<UntypedScalarRedOp<UnaryRedCode::ARGMIN>>(
    first_redop_id + NUMPY_SCALAR_ARGMIN_REDOP);

  // Register our projection and sharding functions
  const ProjectionID first_projection_id =
    runtime->generate_library_projection_ids(numpy_library_name, NUMPY_PROJ_LAST);
  NumPyProjectionFunctor::register_projection_functors(runtime, first_projection_id);
  const ShardingID first_sharding_id =
    runtime->generate_library_sharding_ids(numpy_library_name, NUMPY_SHARD_LAST);
  NumPyShardingFunctor::register_sharding_functors(runtime, first_sharding_id);

  // Now we can register our mapper with the runtime
  const MapperID numpy_mapper_id =
    runtime->generate_library_mapper_ids(numpy_library_name, NUMPY_MAX_MAPPERS);
  // This will register it with all the processors on the node
  runtime->add_mapper(numpy_mapper_id,
                      new NumPyMapper(runtime->get_mapper_runtime(),
                                      machine,
                                      first_tid,
                                      first_tid + max_numpy_tasks - 1,
                                      first_sharding_id));
}

}  // namespace numpy
}  // namespace legate

extern "C" {

void legate_numpy_perform_registration(void)
{
  // Tell the runtime about our registration callback so we hook it
  // in before the runtime starts and make it global so that we know
  // that this call back is invoked everywhere across all nodes
  Runtime::perform_registration_callback(legate::numpy::registration_callback, true /*global*/);
}
}
