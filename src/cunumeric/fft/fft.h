/* Copyright 2022 NVIDIA Corporation
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
#include "cunumeric/fft/fft_util.h"

namespace cunumeric {

struct FFTArgs {
  Array output;
  Array input;
  CuNumericFFTType type;
  CuNumericFFTDirection direction;
  bool operate_over_axes;
  std::vector<int64_t> axes;

  // We assume that all gpus within a task are on the same node
  // AND the id corresponds to their natural order
  // This needs to be ensured by the resource scoping
  int32_t gpu_id;
  int32_t num_gpus;
};

class FFTTask : public CuNumericTask<FFTTask> {
 public:
  static const int TASK_ID = CUNUMERIC_FFT;

 public:
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

}  // namespace cunumeric
