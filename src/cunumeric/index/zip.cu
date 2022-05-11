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
             const Buffer<AccessorRO<int64_t, DIM>, 1> index_arrays,
             const Rect<DIM> rect,
             const Pitches<DIM - 1> pitches,
             size_t volume,
             DomainPoint shape,
             std::index_sequence<Is...>)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto p = pitches.unflatten(idx, rect.lo);
  Legion::Point<N> new_point;
  for (size_t i = 0; i < N; i++) { new_point[i] = compute_idx(index_arrays[i][p], shape[i]); }
  out[p] = new_point;
}

template <int DIM, int N, size_t... Is>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  zip_kernel_dense(Point<N>* out,
                   const Buffer<const int64_t*, 1> index_arrays,
                   const Rect<DIM> rect,
                   size_t volume,
                   DomainPoint shape,
                   std::index_sequence<Is...>)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  Legion::Point<N> new_point;
  for (size_t i = 0; i < N; i++) { new_point[i] = compute_idx(index_arrays[i][idx], shape[i]); }
  out[idx] = new_point;
}

template <int DIM, int N>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  zip_kernel(const AccessorWO<Point<N>, DIM> out,
             const Buffer<AccessorRO<int64_t, DIM>, 1> index_arrays,
             const Rect<DIM> rect,
             const Pitches<DIM - 1> pitches,
             int narrays,
             size_t volume,
             int64_t key_dim,
             int64_t start_index,
             DomainPoint shape)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto p = pitches.unflatten(idx, rect.lo);
  Legion::Point<N> new_point;
  for (size_t i = 0; i < start_index; i++) { new_point[i] = p[i]; }
  for (size_t i = 0; i < narrays; i++) {
    new_point[start_index + i] = compute_idx(index_arrays[i][p], shape[start_index + i]);
  }
  for (size_t i = (start_index + narrays); i < N; i++) {
    int64_t j    = key_dim + i - narrays;
    new_point[i] = p[j];
  }
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
                  const int64_t start_index,
                  const DomainPoint& shape,
                  std::index_sequence<Is...>) const
  {
    auto stream         = get_cached_stream();
    const size_t volume = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (index_arrays.size() == N) {
      if (dense) {
        auto index_buf =
          create_buffer<const int64_t*, 1>(index_arrays.size(), Memory::Kind::Z_COPY_MEM);
        for (uint32_t idx = 0; idx < index_arrays.size(); ++idx) {
          index_buf[idx] = index_arrays[idx].ptr(rect);
        }
        zip_kernel_dense<DIM, N><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          out.ptr(rect), index_buf, rect, volume, shape, std::make_index_sequence<N>());
      } else {
        auto index_buf =
          create_buffer<AccessorRO<VAL, DIM>, 1>(index_arrays.size(), Memory::Kind::Z_COPY_MEM);
        for (uint32_t idx = 0; idx < index_arrays.size(); ++idx) index_buf[idx] = index_arrays[idx];
        zip_kernel<DIM, N><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          out, index_buf, rect, pitches, volume, shape, std::make_index_sequence<N>());
      }
    } else {
#ifdef DEBUG_CUNUMERIC
      assert(index_arrays.size() < N);
#endif
      auto index_buf =
        create_buffer<AccessorRO<VAL, DIM>, 1>(index_arrays.size(), Memory::Kind::Z_COPY_MEM);
      for (uint32_t idx = 0; idx < index_arrays.size(); ++idx) index_buf[idx] = index_arrays[idx];
      int num_arrays = index_arrays.size();
      zip_kernel<DIM, N><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        out, index_buf, rect, pitches, num_arrays, volume, key_dim, start_index, shape);
    }
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void ZipTask::gpu_variant(TaskContext& context)
{
  zip_template<VariantKind::GPU>(context);
}
}  // namespace cunumeric
