/* Copyright 2023 NVIDIA Corporation
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

#include "cunumeric/index/select.h"
#include "cunumeric/index/select_template.inl"
#include "cunumeric/cuda_help.h"

namespace cunumeric {

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  select_kernel_dense(VAL* outptr,
                      uint32_t narrays,
                      legate::Buffer<const bool*, 1> condlist,
                      legate::Buffer<const VAL*, 1> choicelist,
                      VAL default_val,
                      int volume)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) return;
  for (uint32_t c = 0; c < narrays; ++c) {
    if (condlist[c][idx]) {
      outptr[idx] = choicelist[c][idx];
      return;
    }
  }
  outptr[idx] = default_val;
}

}  // namespace cunumeric
