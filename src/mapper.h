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

#pragma once

#include "numpy.h"

#include "mapping/base_mapper.h"

namespace legate {
namespace numpy {

class NumPyMapper : public mapping::BaseMapper {
 public:
  NumPyMapper(Legion::Mapping::MapperRuntime* rt,
              Legion::Machine machine,
              const LibraryContext& context);
  virtual ~NumPyMapper(void) {}

 private:
  NumPyMapper(const NumPyMapper& rhs) = delete;
  NumPyMapper& operator=(const NumPyMapper& rhs) = delete;

  // Legate mapping functions
 public:
  virtual bool is_pure() const override { return true; }
  virtual mapping::TaskTarget task_target(const mapping::Task& task,
                                          const std::vector<mapping::TaskTarget>& options) override;
  virtual std::vector<mapping::StoreMapping> store_mappings(
    const mapping::Task& task, const std::vector<mapping::StoreTarget>& options) override;
  virtual Scalar tunable_value(TunableID tunable_id) override;

 private:
  const int32_t min_gpu_chunk;
  const int32_t min_cpu_chunk;
  const int32_t min_omp_chunk;
  const int32_t eager_fraction;
};

}  // namespace numpy
}  // namespace legate
