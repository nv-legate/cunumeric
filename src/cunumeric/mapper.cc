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

#include "cunumeric/mapper.h"

using namespace legate;
using namespace legate::mapping;

namespace cunumeric {

CuNumericMapper::CuNumericMapper(Legion::Runtime* rt, Legion::Machine m, const LibraryContext& ctx)
  : BaseMapper(rt, m, ctx),
    min_gpu_chunk(extract_env("CUNUMERIC_MIN_GPU_CHUNK", 1 << 20, 2)),
    min_cpu_chunk(extract_env("CUNUMERIC_MIN_CPU_CHUNK", 1 << 14, 2)),
    min_omp_chunk(extract_env("CUNUMERIC_MIN_OMP_CHUNK", 1 << 17, 2)),
    eager_fraction(extract_env("CUNUMERIC_EAGER_FRACTION", 16, 1))
{
}

TaskTarget CuNumericMapper::task_target(const Task& task, const std::vector<TaskTarget>& options)
{
  return *options.begin();
}

Scalar CuNumericMapper::tunable_value(TunableID tunable_id)
{
  switch (tunable_id) {
    case CUNUMERIC_TUNABLE_NUM_GPUS: {
      int32_t num_gpus = local_gpus.size() * total_nodes;
      return Scalar(num_gpus);
    }
    case CUNUMERIC_TUNABLE_NUM_PROCS: {
      int32_t num_procs = 0;
      if (!local_gpus.empty())
        num_procs = local_gpus.size() * total_nodes;
      else if (!local_omps.empty())
        num_procs = local_omps.size() * total_nodes;
      else
        num_procs = local_cpus.size() * total_nodes;
      return Scalar(num_procs);
    }
    case CUNUMERIC_TUNABLE_MAX_EAGER_VOLUME: {
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
    case CUNUMERIC_TUNABLE_HAS_NUMAMEM: {
      // TODO: This assumes that either all OpenMP processors across the machine have a NUMA
      // memory or none does.
      Legion::Machine::MemoryQuery query(machine);
      query.local_address_space();
      query.only_kind(Legion::Memory::SOCKET_MEM);
      int32_t has_numamem = query.count() > 0;
      return Scalar(has_numamem);
    }
    default: break;
  }
  LEGATE_ABORT;  // unknown tunable value
}

std::vector<StoreMapping> CuNumericMapper::store_mappings(
  const mapping::Task& task, const std::vector<mapping::StoreTarget>& options)
{
  switch (task.task_id()) {
    case CUNUMERIC_CONVOLVE: {
      std::vector<StoreMapping> mappings;
      auto& inputs = task.inputs();
      mappings.push_back(StoreMapping::default_mapping(inputs[0], options.front()));
      mappings.push_back(StoreMapping::default_mapping(inputs[1], options.front()));
      auto& input_mapping = mappings.back();
      for (uint32_t idx = 2; idx < inputs.size(); ++idx)
        input_mapping.stores.push_back(inputs[idx]);
      return std::move(mappings);
    }
    case CUNUMERIC_FFT: {
      std::vector<StoreMapping> mappings;
      auto& inputs  = task.inputs();
      auto& outputs = task.outputs();
      mappings.push_back(StoreMapping::default_mapping(inputs[0],  options.front()));
      mappings.push_back(StoreMapping::default_mapping(outputs[0], options.front()));
      return std::move(mappings);
    }
    case CUNUMERIC_TRANSPOSE_COPY_2D: {
      auto logical = task.scalars()[0].value<bool>();
      if (!logical) {
        std::vector<StoreMapping> mappings;
        auto& outputs = task.outputs();
        mappings.push_back(StoreMapping::default_mapping(outputs[0], options.front()));
        mappings.back().policy.ordering.fortran_order();
        mappings.back().policy.exact = true;
        return std::move(mappings);
      } else
        return {};
    }
    case CUNUMERIC_MATMUL:
    case CUNUMERIC_MATVECMUL: {
      // TODO: Our actual requirements are a little less strict than this; we require each array or
      // vector to have a stride of 1 on at least one dimension.
      std::vector<StoreMapping> mappings;
      auto& inputs  = task.inputs();
      auto& outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(StoreMapping::default_mapping(input, options.front()));
        mappings.back().policy.exact = true;
      }
      for (auto& output : outputs) {
        mappings.push_back(StoreMapping::default_mapping(output, options.front()));
        mappings.back().policy.exact = true;
      }
      return std::move(mappings);
    }
    case CUNUMERIC_POTRF:
    case CUNUMERIC_TRSM:
    case CUNUMERIC_SYRK:
    case CUNUMERIC_GEMM: {
      std::vector<StoreMapping> mappings;
      auto& inputs  = task.inputs();
      auto& outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(StoreMapping::default_mapping(input, options.front()));
        mappings.back().policy.ordering.fortran_order();
        mappings.back().policy.exact = true;
      }
      for (auto& output : outputs) {
        mappings.push_back(StoreMapping::default_mapping(output, options.front()));
        mappings.back().policy.ordering.fortran_order();
        mappings.back().policy.exact = true;
      }
      return std::move(mappings);
    }
    case CUNUMERIC_TRILU: {
      if (task.scalars().size() == 2) return {};
      // If we're here, this task was the post-processing for Cholesky.
      // So we will request fortran ordering
      std::vector<StoreMapping> mappings;
      auto& input = task.inputs().front();
      mappings.push_back(StoreMapping::default_mapping(input, options.front()));
      mappings.back().policy.ordering.fortran_order();
      mappings.back().policy.exact = true;
      return std::move(mappings);
    }
    case CUNUMERIC_SORT: {
      std::vector<StoreMapping> mappings;
      auto& inputs = task.inputs();
      mappings.push_back(StoreMapping::default_mapping(inputs[0], options.front()));
      mappings.back().policy.ordering.c_order();
      mappings.back().policy.exact = true;
      return std::move(mappings);
    }
    default: {
      return {};
    }
  }
  assert(false);
  return {};
}

}  // namespace cunumeric
