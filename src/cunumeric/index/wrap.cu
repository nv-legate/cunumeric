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

#include "cunumeric/index/wrap.h"
#include "cunumeric/index/wrap_template.inl"
#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  wrap_kernel(const AccessorWO<Point<DIM>, 1> out,
              const int64_t start,
              const int64_t volume,
              const Pitches<0> pitches_out,
              const Point<1> out_lo,
              const Pitches<DIM - 1> pitches_in,
              const Point<DIM> in_lo,
              const size_t in_volume)
{
  const auto idx = global_tid_1d();
  if (idx >= volume) return;
  const int64_t input_idx = (idx + start) % in_volume;
  auto out_p              = pitches_out.unflatten(idx, out_lo);
  auto p                  = pitches_in.unflatten(input_idx, in_lo);
  out[out_p]              = p;
}

template <int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  wrap_kernel_dense(Point<DIM>* out,
                    const int64_t start,
                    const int64_t volume,
                    const Pitches<DIM - 1> pitches_in,
                    const Point<DIM> in_lo,
                    const size_t in_volume)
{
  const auto idx = global_tid_1d();
  if (idx >= volume) return;
  const int64_t input_idx = (idx + start) % in_volume;
  auto p                  = pitches_in.unflatten(input_idx, in_lo);
  out[idx]                = p;
}

template <int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  wrap_kernel(const AccessorWO<Point<DIM>, 1> out,
              const int64_t start,
              const int64_t volume,
              const Pitches<0> pitches_out,
              const Point<1> out_lo,
              const Pitches<DIM - 1> pitches_in,
              const Point<DIM> in_lo,
              const AccessorRO<int64_t, 1> indices)
{
  const auto idx = global_tid_1d();
  if (idx >= volume) return;
  auto out_p              = pitches_out.unflatten(idx, out_lo);
  const int64_t input_idx = indices[out_p];
  auto p                  = pitches_in.unflatten(input_idx, in_lo);
  out[out_p]              = p;
}

template <int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  wrap_kernel_dense(Point<DIM>* out,
                    const int64_t start,
                    const int64_t volume,
                    const Pitches<DIM - 1> pitches_in,
                    const Point<DIM> in_lo,
                    const AccessorRO<int64_t, 1> indices)
{
  const auto idx = global_tid_1d();
  if (idx >= volume) return;
  const int64_t input_idx = indices[idx];
  auto p                  = pitches_in.unflatten(input_idx, in_lo);
  out[idx]                = p;
}

template <int DIM>
struct WrapImplBody<VariantKind::GPU, DIM> {
  void operator()(const AccessorWO<Point<DIM>, 1>& out,
                  const Pitches<0>& pitches_out,
                  const Rect<1>& out_rect,
                  const Pitches<DIM - 1>& pitches_in,
                  const Rect<DIM>& in_rect,
                  const bool dense) const
  {
    auto stream          = get_cached_stream();
    const int64_t start  = out_rect.lo[0];
    const int64_t volume = out_rect.volume();
    const auto in_volume = in_rect.volume();
    const size_t blocks  = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dense) {
      auto outptr = out.ptr(out_rect);
      wrap_kernel_dense<DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        outptr, start, volume, pitches_in, in_rect.lo, in_volume);
    } else {
      wrap_kernel<DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        out, start, volume, pitches_out, out_rect.lo, pitches_in, in_rect.lo, in_volume);
    }
    CHECK_CUDA_STREAM(stream);
  }
  void operator()(const AccessorWO<Point<DIM>, 1>& out,
                  const Pitches<0>& pitches_out,
                  const Rect<1>& out_rect,
                  const Pitches<DIM - 1>& pitches_in,
                  const Rect<DIM>& in_rect,
                  const bool dense,
                  const AccessorRO<int64_t, 1>& indices) const
  {
    auto stream          = get_cached_stream();
    const int64_t start  = out_rect.lo[0];
    const int64_t volume = out_rect.volume();
    const size_t blocks  = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dense) {
      auto outptr = out.ptr(out_rect);
      wrap_kernel_dense<DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        outptr, start, volume, pitches_in, in_rect.lo, indices);
    } else {
      wrap_kernel<DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        out, start, volume, pitches_out, out_rect.lo, pitches_in, in_rect.lo, indices);
    }
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void WrapTask::gpu_variant(TaskContext& context)
{
  wrap_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
