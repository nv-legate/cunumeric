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

#include "cunumeric.h"
#include "mapper.h"
#include "unary/unary_red_util.h"

using namespace Legion;
using namespace legate;

namespace cunumeric {

static const char* const cunumeric_library_name = "cunumeric";

/*static*/ bool CuNumeric::has_numamem   = false;
/*static*/ MapperID CuNumeric::mapper_id = -1;

/*static*/ LegateTaskRegistrar& CuNumeric::get_registrar()
{
  static LegateTaskRegistrar registrar;
  return registrar;
}

#ifdef LEGATE_USE_CUDA
extern void register_gpu_reduction_operators(LibraryContext& context);
#else
extern void register_cpu_reduction_operators(LibraryContext& context);
#endif

void registration_callback(Machine machine,
                           Runtime* runtime,
                           const std::set<Processor>& local_procs)
{
  ResourceConfig config;
  config.max_mappers       = CUNUMERIC_MAX_MAPPERS;
  config.max_tasks         = CUNUMERIC_MAX_TASKS;
  config.max_reduction_ops = CUNUMERIC_MAX_REDOPS;
  LibraryContext context(runtime, cunumeric_library_name, config);

  CuNumeric::get_registrar().register_all_tasks(runtime, context);

  // Register our special reduction functions
#ifdef LEGATE_USE_CUDA
  register_gpu_reduction_operators(context);
#else
  register_cpu_reduction_operators(context);
#endif

  // Now we can register our mapper with the runtime
  CuNumeric::mapper_id = context.get_mapper_id(0);
  auto mapper          = new CuNumericMapper(runtime, machine, context);
  // This will register it with all the processors on the node
  runtime->add_mapper(CuNumeric::mapper_id, mapper);
}

}  // namespace cunumeric

extern "C" {

void cunumeric_perform_registration(void)
{
  // Tell the runtime about our registration callback so we hook it
  // in before the runtime starts and make it global so that we know
  // that this call back is invoked everywhere across all nodes
  Runtime::perform_registration_callback(cunumeric::registration_callback, true /*global*/);

  Runtime* runtime = Runtime::get_runtime();
  Context ctx      = Runtime::get_context();
  Future fut       = runtime->select_tunable_value(
    ctx, CUNUMERIC_TUNABLE_HAS_NUMAMEM, cunumeric::CuNumeric::mapper_id);
  if (fut.get_result<int32_t>() != 0) cunumeric::CuNumeric::has_numamem = true;
}
}
