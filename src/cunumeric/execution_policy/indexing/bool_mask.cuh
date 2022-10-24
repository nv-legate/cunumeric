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
#include "cunumeric/execution_policy/indexing/bool_mask.h"
#include "cunumeric/cuda_help.h"

namespace cunumeric {

template <typename T, class KERNEL>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  bool_mask_dense_kernel(const size_t volume, T* maskptr, KERNEL kernel)
{
  const auto idx = global_tid_1d();
  if (idx >= volume) return;
  if (maskptr[idx]) kernel(idx);
}

template <class RECT, class PITCHES, class ACC, class KERNEL>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) bool_mask_kernel(
  const size_t volume, const RECT rect, const PITCHES pitches, ACC mask, KERNEL kernel)
{
  const auto idx = global_tid_1d();
  if (idx >= volume) return;
  auto p = pitches.unflatten(idx, rect.lo);
  if (mask[p]) kernel(p);
}

template <>
struct BoolMaskPolicy<VariantKind::GPU, true> {
  template <class RECT, class ACC, class KERNEL>
  void operator()(const RECT& rect, const ACC& mask, KERNEL&& kernel)
  {
    const size_t volume = rect.volume();
    if (0 == volume) return;
    auto stream         = get_cached_stream();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto maskptr        = mask.ptr(rect);

    bool_mask_dense_kernel<<<blocks, THREADS_PER_BLOCK, 1, stream>>>(
      volume, maskptr, std::forward<KERNEL>(kernel));

    CHECK_CUDA_STREAM(stream);
  }
};

template <>
struct BoolMaskPolicy<VariantKind::GPU, false> {
  template <class RECT, class PITCHES, class ACC, class KERNEL>
  void operator()(const RECT& rect, const PITCHES& pitches, const ACC& mask, KERNEL&& kernel)
  {
    const size_t volume = rect.volume();
    if (0 == volume) return;
    auto stream         = get_cached_stream();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    bool_mask_kernel<<<blocks, THREADS_PER_BLOCK, 1, stream>>>(
      volume, rect, pitches, mask, std::forward<KERNEL>(kernel));
    CHECK_CUDA_STREAM(stream);
  }
};

}  // namespace cunumeric
