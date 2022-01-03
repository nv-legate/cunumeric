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

#include "cunumeric/index/choose.h"
#include "cunumeric/index/choose_template.inl"
#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  choose_kernel(const AccessorWO<VAL, DIM> out,
                           const AccessorRO<int64_t, DIM> index_arr,
                           const DeferredBuffer<AccessorRO<VAL, DIM>, 1> choices,
                           const Rect<DIM> rect,
                           const Pitches<DIM - 1> pitches,
                           int volume)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto p = pitches.unflatten(idx, rect.lo);
  out[p] = choices[index_arr[p]][p];
}

//dense version
template <typename VAL, int DIM, typename OUT, typename IN>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  choose_kernel_dense( OUT* outptr,
                           const IN*  indexptr,
                           const DeferredBuffer<AccessorRO<VAL, DIM>, 1> choices,
                           const Rect<DIM> rect,
                           int volume)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto chptr  = choices[indexptr[idx]].ptr(rect); 
  outptr[idx] = chptr[idx];
}

template <LegateTypeCode CODE, int DIM>
struct ChooseImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(const AccessorWO<VAL, DIM>& out,
                  const AccessorRO<int64_t, DIM>& index_arr,
                  const std::vector<AccessorRO<VAL, DIM>>& choices,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    DeferredBuffer<AccessorRO<VAL, DIM>, 1> ch_arr(Memory::Kind::Z_COPY_MEM,
      Rect<1>(0, choices.size() - 1));
    for (uint32_t idx = 0; idx < choices.size(); ++idx)
      ch_arr[idx] = choices[idx];

    if (dense){
      auto outptr   = out.ptr(rect);
      auto indexptr = index_arr.ptr(rect);
      choose_kernel_dense<VAL, DIM><<<blocks, THREADS_PER_BLOCK>>>(outptr,
        indexptr, ch_arr, rect, volume);
    }else{
      choose_kernel<VAL, DIM><<<blocks, THREADS_PER_BLOCK>>>(out,
        index_arr, ch_arr, rect, pitches, volume);
    }
  }
};

/*static*/ void ChooseTask::gpu_variant(TaskContext& context)
{
  choose_template<VariantKind::GPU>(context);
}
}  // namespace cunumeric
