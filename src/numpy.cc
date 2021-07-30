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
#include "unary/unary_red_util.h"

using namespace Legion;

namespace legate {
namespace numpy {

static const char* const numpy_library_name = "legate.numpy";

/*static*/ LegateTaskRegistrar& LegateNumPy::get_registrar()
{
  static LegateTaskRegistrar registrar;
  return registrar;
}

#ifdef LEGATE_USE_CUDA
extern void register_gpu_reduction_operators(LibraryContext& context);
#else
extern void register_cpu_reduction_operators(LibraryContext& context);
#endif

void register_reduction_operators_for_untyped_scalar(LibraryContext& context)
{
#define REGISTER(OP)                                                    \
  Runtime::register_reduction_op<UntypedScalarRedOp<UnaryRedCode::OP>>( \
    context.get_reduction_op_id(NUMPY_SCALAR_##OP##_REDOP));

  REGISTER(MAX)
  REGISTER(MIN)
  REGISTER(PROD)
  REGISTER(SUM)
  REGISTER(ARGMAX)
  REGISTER(ARGMIN)

#undef REGISTER
}

void registration_callback(Machine machine,
                           Runtime* runtime,
                           const std::set<Processor>& local_procs)
{
  ResourceConfig config;
  config.max_mappers       = NUMPY_MAX_MAPPERS;
  config.max_tasks         = NUMPY_MAX_TASKS;
  config.max_reduction_ops = NUMPY_MAX_REDOPS;
  LibraryContext context(runtime, numpy_library_name, config);

  LegateNumPy::get_registrar().register_all_tasks(runtime, context);

  // Register our special reduction functions
#ifdef LEGATE_USE_CUDA
  register_gpu_reduction_operators(context);
#else
  register_cpu_reduction_operators(context);
#endif

  register_reduction_operators_for_untyped_scalar(context);

  // Now we can register our mapper with the runtime
  auto numpy_mapper_id = context.get_mapper_id(0);
  auto mapper          = new NumPyMapper(runtime->get_mapper_runtime(), machine, context);
  // This will register it with all the processors on the node
  runtime->add_mapper(numpy_mapper_id, mapper);
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
