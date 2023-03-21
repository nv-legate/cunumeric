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

#pragma once

#include "cunumeric/cunumeric.h"

#include "core/mapping/base_mapper.h"

namespace cunumeric {

class CuNumericMapper : public legate::mapping::LegateMapper {
 public:
  CuNumericMapper();
  virtual ~CuNumericMapper(void) {}

 private:
  CuNumericMapper(const CuNumericMapper& rhs)            = delete;
  CuNumericMapper& operator=(const CuNumericMapper& rhs) = delete;

  // Legate mapping functions
 public:
  virtual void set_machine(const legate::mapping::MachineQueryInterface* machine) override;
  virtual legate::mapping::TaskTarget task_target(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::TaskTarget>& options) override;
  virtual std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override;
  virtual legate::Scalar tunable_value(legate::TunableID tunable_id) override;

 private:
  const legate::mapping::MachineQueryInterface* machine;
  const int32_t min_gpu_chunk;
  const int32_t min_cpu_chunk;
  const int32_t min_omp_chunk;
  const int32_t eager_fraction;
};

}  // namespace cunumeric
