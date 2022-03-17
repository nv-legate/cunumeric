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

#include "cunumeric/index/zip.h"
#include "cunumeric/index/zip_template.inl"
#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;

template <int DIM, int N, size_t... Is>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  zip_kernel(const AccessorWO<Point<N>, DIM> out,
             const DeferredBuffer<AccessorRO<int64_t, DIM>, 1> index_arrays,
             const Rect<DIM> rect,
             const Pitches<DIM - 1> pitches,
             int volume,
             std::index_sequence<Is...>)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto p = pitches.unflatten(idx, rect.lo);
  out[p] = Legion::Point<N>(index_arrays[Is][p]...);
}

template <int DIM, int N, size_t... Is>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  zip_kernel_dense(Point<N>* out,
                   const DeferredBuffer<const int64_t*, 1> index_arrays,
                   const Rect<DIM> rect,
                   int volume,
                   std::index_sequence<Is...>)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  out[idx] = Legion::Point<N>(index_arrays[Is][idx]...);
}

template <int DIM, int N>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  zip_kernel(const AccessorWO<Point<N>, DIM> out,
             const AccessorRO<int64_t, DIM> index_array,
             const Rect<DIM> rect,
             const Pitches<DIM - 1> pitches,
             int volume,
             const int64_t key_dim)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto p = pitches.unflatten(idx, rect.lo);
  Legion::Point<N> new_point;
  new_point[0] = index_array[p];
  for (size_t i = 1; i < N; i++) { new_point[i] = p[key_dim + i - 1]; }
  out[p] = new_point;
}

template <int DIM, int N>
struct ZipImplBody<VariantKind::GPU, DIM, N> {
  using VAL = int64_t;

  template <size_t... Is>
  void operator()(const AccessorWO<Point<N>, DIM>& out,
                  const std::vector<AccessorRO<VAL, DIM>>& index_arrays,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  bool dense,
                  const int64_t key_dim,
                  std::index_sequence<Is...>) const
  {
    const size_t volume = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (index_arrays.size() > 1) {
      if (dense) {
        DeferredBuffer<const int64_t*, 1> idx_arr(Memory::Kind::Z_COPY_MEM,
                                                  Rect<1>(0, index_arrays.size() - 1));
        for (uint32_t idx = 0; idx < index_arrays.size(); ++idx) {
          idx_arr[idx] = index_arrays[idx].ptr(rect);
        }
        zip_kernel_dense<DIM, N><<<blocks, THREADS_PER_BLOCK>>>(
          out.ptr(rect), idx_arr, rect, volume, std::make_index_sequence<N>());
      } else {
        DeferredBuffer<AccessorRO<VAL, DIM>, 1> idx_arr(Memory::Kind::Z_COPY_MEM,
                                                        Rect<1>(0, index_arrays.size() - 1));
        for (uint32_t idx = 0; idx < index_arrays.size(); ++idx) idx_arr[idx] = index_arrays[idx];
        zip_kernel<DIM, N><<<blocks, THREADS_PER_BLOCK>>>(
          out, idx_arr, rect, pitches, volume, std::make_index_sequence<N>());
      }
    } else if (index_arrays.size() == 1) {
      zip_kernel<DIM, N>
        <<<blocks, THREADS_PER_BLOCK>>>(out, index_arrays[0], rect, pitches, volume, key_dim);
    }
  }
};

/*static*/ void ZipTask::gpu_variant(TaskContext& context)
{
  zip_template<VariantKind::GPU>(context);
}
}  // namespace cunumeric
