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
#include "cunumeric/cuda_help.h"
#include "cunumeric/execution_policy/indexing/replace_nan.h"

namespace cunumeric {

template <class KERNEL, class LHS, class Tag>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  replace_nan_kernel(const size_t volume, LHS identity, KERNEL kernel, Tag tag)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) return;
  auto value = identity;
  kernel(idx, value, tag);
}

template <class Tag>
struct ReplaceNanPolicy<VariantKind::GPU, Tag> {
  template <class RECT, class LHS, class KERNEL>
  void operator()(const RECT& rect, LHS& identity, KERNEL&& kernel)
  {
    const size_t volume = rect.volume();
    if (0 == volume) return;
    auto stream         = get_cached_stream();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    replace_nan_kernel<<<blocks, THREADS_PER_BLOCK, 1, stream>>>(
      volume, identity, std::forward<KERNEL>(kernel), Tag{});

    CHECK_CUDA_STREAM(stream);
  }
};

}  // namespace cunumeric
