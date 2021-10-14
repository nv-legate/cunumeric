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

#include "numpy/mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

namespace legate {

using namespace mapping;

namespace numpy {

NumPyMapper::NumPyMapper(MapperRuntime* rt, Machine m, const LibraryContext& ctx)
  : BaseMapper(rt, m, ctx),
    min_gpu_chunk(extract_env("NUMPY_MIN_GPU_CHUNK", 1 << 20, 2)),
    min_cpu_chunk(extract_env("NUMPY_MIN_CPU_CHUNK", 1 << 14, 2)),
    min_omp_chunk(extract_env("NUMPY_MIN_OMP_CHUNK", 1 << 17, 2)),
    eager_fraction(extract_env("NUMPY_EAGER_FRACTION", 16, 1))
{
}

TaskTarget NumPyMapper::task_target(const Task& task, const std::vector<TaskTarget>& options)
{
  return *options.begin();
}

Scalar NumPyMapper::tunable_value(TunableID tunable_id)
{
  switch (tunable_id) {
    case NUMPY_TUNABLE_NUM_GPUS: {
      int32_t num_gpus = local_gpus.empty() ? 0 : local_gpus.size() * total_nodes;
      return Scalar(num_gpus);
    }
    case NUMPY_TUNABLE_MAX_EAGER_VOLUME: {
      int32_t eager_volume = 0;
      // TODO: make these profile guided
      if (eager_fraction > 0) {
        if (!local_gpus.empty())
          eager_volume = min_gpu_chunk / eager_fraction;
        else if (!local_omps.empty())
          eager_volume = min_omp_chunk / eager_fraction;
        else
          eager_volume = min_cpu_chunk / eager_fraction;
      }
      return Scalar(eager_volume);
    }
    default: break;
  }
  LEGATE_ABORT  // unknown tunable value
}

std::vector<StoreMapping> NumPyMapper::store_mappings(
  const mapping::Task& task, const std::vector<mapping::StoreTarget>& options)
{
  return {};
}

}  // namespace numpy
}  // namespace legate
