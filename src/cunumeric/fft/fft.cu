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
#include <csignal>

#include "cunumeric/fft/fft.h"
#include "cunumeric/fft/fft_template.inl"

#include "cunumeric/cuda_help.h"
#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace legate;

using dim_t = long long int32_t;

template <int32_t DIM, typename TYPE>
__global__ static void copy_kernel(size_t volume,
                                   Buffer<TYPE, DIM> buffer,
                                   AccessorRO<TYPE, DIM> acc,
                                   Pitches<DIM - 1> pitches,
                                   Point<DIM> lo)
{
  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= volume) return;
  auto p    = pitches.unflatten(offset, Point<DIM>::ZEROES());
  buffer[p] = acc[p + lo];
}

template <int32_t DIM, typename TYPE>
__host__ static inline void copy_into_buffer(Buffer<TYPE, DIM>& buffer,
                                             AccessorRO<TYPE, DIM>& acc,
                                             const Rect<DIM>& rect,
                                             size_t volume,
                                             cudaStream_t stream)
{
  if (acc.accessor.is_dense_row_major(rect)) {
    auto zero = Point<DIM>::ZEROES();
    CHECK_CUDA(cudaMemcpyAsync(
      buffer.ptr(zero), acc.ptr(zero), volume * sizeof(TYPE), cudaMemcpyDefault, stream));
  } else {
    Pitches<DIM - 1> pitches;
    pitches.flatten(rect);

    const size_t num_blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    copy_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      volume, buffer, acc, pitches, rect.lo);

    CHECK_CUDA_STREAM(stream);
  }
}

template <int32_t DIM, typename OUTPUT_TYPE, typename INPUT_TYPE>
__host__ static inline void cufft_operation(AccessorWO<OUTPUT_TYPE, DIM> out,
                                            AccessorRO<INPUT_TYPE, DIM> in,
                                            const Rect<DIM>& out_rect,
                                            const Rect<DIM>& in_rect,
                                            std::vector<int64_t>& axes,
                                            CuNumericFFTType type,
                                            CuNumericFFTDirection direction)
{
  auto stream = get_cached_stream();

  size_t workarea_size = 0;
  size_t num_elements;
  dim_t n[DIM];
  dim_t inembed[DIM];
  dim_t onembed[DIM];

  const Point<DIM> zero   = Point<DIM>::ZEROES();
  const Point<DIM> one    = Point<DIM>::ONES();
  Point<DIM> fft_size_in  = in_rect.hi - in_rect.lo + one;
  Point<DIM> fft_size_out = out_rect.hi - out_rect.lo + one;
  num_elements            = 1;
  for (int32_t i = 0; i < DIM; ++i) {
    n[i] =
      (type == CUNUMERIC_FFT_R2C || type == CUNUMERIC_FFT_D2Z) ? fft_size_in[i] : fft_size_out[i];
    inembed[i] = fft_size_in[i];
    onembed[i] = fft_size_out[i];
    num_elements *= n[i];
  }

  // Create the plan
  cufftHandle plan;
  CHECK_CUFFT(cufftCreate(&plan));
  CHECK_CUFFT(cufftSetAutoAllocation(plan, 0 /*we'll do the allocation*/));
  CHECK_CUFFT(cufftSetStream(plan, stream));

  // Create the plan and allocate a temporary buffer for it if it needs one
  CHECK_CUFFT(cufftMakePlanMany64(
    plan, DIM, n, inembed, 1, 1, onembed, 1, 1, static_cast<cufftType>(type), 1, &workarea_size));

  if (workarea_size > 0) {
    auto workarea_buffer = create_buffer<uint8_t>(workarea_size, Memory::Kind::GPU_FB_MEM);
    CHECK_CUFFT(cufftSetWorkArea(plan, workarea_buffer.ptr(0)));
  }

  const void* in_ptr{nullptr};
  if (in.accessor.is_dense_row_major(in_rect))
    in_ptr = in.ptr(in_rect.lo);
  else {
    auto buffer = create_buffer<INPUT_TYPE, DIM>(fft_size_in, Memory::Kind::GPU_FB_MEM);
    copy_into_buffer(buffer, in, in_rect, in_rect.volume(), stream);
    in_ptr = buffer.ptr(zero);
  }
  // FFT the input data
  CHECK_CUFFT(cufftXtExec(plan,
                          const_cast<void*>(in_ptr),
                          static_cast<void*>(out.ptr(out_rect.lo)),
                          static_cast<int32_t>(direction)));

  // Clean up our resources, Buffers are cleaned up by Legate
  CHECK_CUFFT(cufftDestroy(plan));
}

template <int32_t DIM, typename OUTPUT, typename INPUT_TYPE>
struct cufft_axes_plan {
  __host__ static inline void execute(cufftHandle plan,
                                      OUTPUT& out,
                                      Buffer<INPUT_TYPE, DIM>& in,
                                      const Rect<DIM>& out_rect,
                                      const Rect<DIM>& in_rect,
                                      int32_t axis,
                                      CuNumericFFTDirection direction)
  {
    const auto zero = Point<DIM>::ZEROES();
    CHECK_CUFFT(cufftXtExec(plan,
                            static_cast<void*>(in.ptr(zero)),
                            static_cast<void*>(out.ptr(out_rect.lo)),
                            static_cast<int32_t>(direction)));
  }
};

// For dimensions higher than 2D, we need to iterate through the input volume as 2D slices due to
// limitations of cuFFT indexing in 1D
template <typename OUTPUT, typename INPUT_TYPE>
struct cufft_axes_plan<3, OUTPUT, INPUT_TYPE> {
  __host__ static inline void execute(cufftHandle plan,
                                      OUTPUT& out,
                                      Buffer<INPUT_TYPE, 3>& in,
                                      const Rect<3>& out_rect,
                                      const Rect<3>& in_rect,
                                      int32_t axis,
                                      CuNumericFFTDirection direction)
  {
    bool is_inner_axis = (axis == 1);
    if (is_inner_axis) {
      // TODO: use PointInRectIterator<DIM>
      auto num_slices = in_rect.hi[0] - in_rect.lo[0] + 1;
      for (uint32_t n = 0; n < num_slices; ++n) {
        const auto offset = Point<3>(n, 0, 0);
        CHECK_CUFFT(cufftXtExec(plan,
                                static_cast<void*>(in.ptr(offset)),
                                static_cast<void*>(out.ptr(out_rect.lo + offset)),
                                static_cast<int32_t>(direction)));
      }
    } else {
      const auto zero = Point<3>::ZEROES();
      CHECK_CUFFT(cufftXtExec(plan,
                              static_cast<void*>(in.ptr(zero)),
                              static_cast<void*>(out.ptr(out_rect.lo)),
                              static_cast<int32_t>(direction)));
    }
  }
};

// Perform the FFT operation as multiple 1D FFTs along the specified axes (Complex-to-complex case).
// For now, it only supports up to 3D FFTs, but final plan is having support for
// N-dimensional FFTs using this approach.
// See cufft_over_axis_r2c_c2r for the equivalent on a single R2C/C2R axis.
template <int32_t DIM, typename OUTPUT_TYPE, typename INPUT_TYPE>
__host__ static inline void cufft_over_axes_c2c(AccessorWO<OUTPUT_TYPE, DIM> out,
                                                AccessorRO<INPUT_TYPE, DIM> in,
                                                const Rect<DIM>& out_rect,
                                                const Rect<DIM>& in_rect,
                                                std::vector<int64_t>& axes,
                                                CuNumericFFTType type,
                                                CuNumericFFTDirection direction)
{
  auto stream = get_cached_stream();

  size_t workarea_size = 0;
  dim_t n[DIM];
  dim_t inembed[DIM];
  dim_t onembed[DIM];

  // Full volume dimensions / strides
  const Point<DIM> zero   = Point<DIM>::ZEROES();
  const Point<DIM> one    = Point<DIM>::ONES();
  Point<DIM> fft_size_in  = in_rect.hi - in_rect.lo + one;
  Point<DIM> fft_size_out = out_rect.hi - out_rect.lo + one;
  size_t num_elements_in  = 1;
  size_t num_elements_out = 1;
  for (int32_t i = 0; i < DIM; ++i) {
    n[i]       = fft_size_out[i];
    inembed[i] = fft_size_in[i];
    onembed[i] = fft_size_out[i];
    num_elements_in *= fft_size_in[i];
    num_elements_out *= fft_size_out[i];
  }

  // Copy input to temporary buffer to perform FFTs one by one
  auto input_buffer = create_buffer<INPUT_TYPE, DIM>(fft_size_in, Memory::Kind::GPU_FB_MEM);
  copy_into_buffer<DIM, INPUT_TYPE>(input_buffer, in, in_rect, num_elements_in, stream);

  Buffer<uint8_t> workarea_buffer;
  size_t last_workarea_size = 0;
  for (auto& ax : axes) {
    // Create the plan
    cufftHandle plan;
    CHECK_CUFFT(cufftCreate(&plan));
    CHECK_CUFFT(cufftSetAutoAllocation(plan, 0 /*we'll do the allocation*/));
    CHECK_CUFFT(cufftSetStream(plan, stream));

    // Single axis dimensions / stridfes
    dim_t size_1d = n[ax];
    // TODO: batches only correct for DIM <= 3. Fix for N-DIM case
    dim_t batches = (DIM == 3 && ax == 1) ? n[2] : num_elements_in / n[ax];
    dim_t istride = 1;
    dim_t ostride = 1;
    for (int32_t i = ax + 1; i < DIM; ++i) {
      istride *= fft_size_in[i];
      ostride *= fft_size_out[i];
    }
    dim_t idist = (ax == DIM - 1) ? fft_size_in[ax] : 1;
    dim_t odist = (ax == DIM - 1) ? fft_size_out[ax] : 1;

    // Create the plan and allocate a temporary buffer for it if it needs one
    CHECK_CUFFT(cufftMakePlanMany64(plan,
                                    1,
                                    &size_1d,
                                    inembed,
                                    istride,
                                    idist,
                                    onembed,
                                    ostride,
                                    odist,
                                    (cufftType)type,
                                    batches,
                                    &workarea_size));

    if (workarea_size > 0) {
      if (workarea_size > last_workarea_size) {
        if (last_workarea_size > 0) workarea_buffer.destroy();
        workarea_buffer    = create_buffer<uint8_t>(workarea_size, Memory::Kind::GPU_FB_MEM);
        last_workarea_size = workarea_size;
      }
      CHECK_CUFFT(cufftSetWorkArea(plan, workarea_buffer.ptr(0)));
    }

    // TODO: following function only correct for DIM <= 3. Fix for N-DIM case
    cufft_axes_plan<DIM, Buffer<INPUT_TYPE, DIM>, INPUT_TYPE>::execute(
      plan, input_buffer, input_buffer, out_rect, in_rect, ax, direction);

    // Clean up our resources, Buffers are cleaned up by Legate
    CHECK_CUFFT(cufftDestroy(plan));
  }
  CHECK_CUDA(cudaMemcpyAsync(out.ptr(zero),
                             input_buffer.ptr(zero),
                             num_elements_out * sizeof(OUTPUT_TYPE),
                             cudaMemcpyDefault,
                             stream));
}

// Perform the FFT operation as multiple 1D FFTs along the specified axes, single R2C/C2R operation.
template <int32_t DIM, typename OUTPUT_TYPE, typename INPUT_TYPE>
__host__ static inline void cufft_over_axis_r2c_c2r(AccessorWO<OUTPUT_TYPE, DIM> out,
                                                    AccessorRO<INPUT_TYPE, DIM> in,
                                                    const Rect<DIM>& out_rect,
                                                    const Rect<DIM>& in_rect,
                                                    std::vector<int64_t>& axes,
                                                    CuNumericFFTType type,
                                                    CuNumericFFTDirection direction)
{
  auto stream = get_cached_stream();

  size_t workarea_size = 0;
  dim_t n[DIM];
  dim_t inembed[DIM];
  dim_t onembed[DIM];

  // Full volume dimensions / strides
  const Point<DIM> zero   = Point<DIM>::ZEROES();
  const Point<DIM> one    = Point<DIM>::ONES();
  Point<DIM> fft_size_in  = in_rect.hi - in_rect.lo + one;
  Point<DIM> fft_size_out = out_rect.hi - out_rect.lo + one;
  size_t num_elements_in  = 1;
  size_t num_elements_out = 1;
  for (int32_t i = 0; i < DIM; ++i) {
    n[i]       = (direction == CUNUMERIC_FFT_FORWARD) ? fft_size_in[i] : fft_size_out[i];
    inembed[i] = fft_size_in[i];
    onembed[i] = fft_size_out[i];
    num_elements_in *= fft_size_in[i];
    num_elements_out *= fft_size_out[i];
  }

  // cuFFT out-of-place C2R always overwrites the input buffer,
  // which is not what we want here, so copy
  // Copy input to temporary buffer to perform FFTs one by one
  auto input_buffer = create_buffer<INPUT_TYPE, DIM>(fft_size_in, Memory::Kind::GPU_FB_MEM);
  copy_into_buffer<DIM, INPUT_TYPE>(input_buffer, in, in_rect, num_elements_in, stream);

  // Create the plan
  cufftHandle plan;
  CHECK_CUFFT(cufftCreate(&plan));
  CHECK_CUFFT(cufftSetAutoAllocation(plan, 0 /*we'll do the allocation*/));
  CHECK_CUFFT(cufftSetStream(plan, stream));

  // Operate over the R2C or C2R axis, which should be the only one in the list
  assert(axes.size() == 1);
  auto axis = axes.front();

  // Batched 1D dimension
  dim_t size_1d = n[axis];
  // TODO: batch only correct for DIM <= 3. Fix for N-DIM case
  dim_t batches = (direction == CUNUMERIC_FFT_FORWARD) ? num_elements_in : num_elements_out;
  batches       = (DIM == 3 && axis == 1) ? n[2] : batches / n[axis];
  dim_t istride = 1;
  dim_t ostride = 1;
  for (int32_t i = axis + 1; i < DIM; ++i) {
    istride *= fft_size_in[i];
    ostride *= fft_size_out[i];
  }
  dim_t idist = (axis == DIM - 1) ? fft_size_in[axis] : 1;
  dim_t odist = (axis == DIM - 1) ? fft_size_out[axis] : 1;

  // Create the plan and allocate a temporary buffer for it if it needs one
  CHECK_CUFFT(cufftMakePlanMany64(plan,
                                  1,
                                  &size_1d,
                                  inembed,
                                  istride,
                                  idist,
                                  onembed,
                                  ostride,
                                  odist,
                                  (cufftType)type,
                                  batches,
                                  &workarea_size));

  if (workarea_size > 0) {
    auto workarea_buffer = create_buffer<uint8_t>(workarea_size, Memory::Kind::GPU_FB_MEM);
    CHECK_CUFFT(cufftSetWorkArea(plan, workarea_buffer.ptr(0)));
  }

  cufft_axes_plan<DIM, AccessorWO<OUTPUT_TYPE, DIM>, INPUT_TYPE>::execute(
    plan, out, input_buffer, out_rect, in_rect, axis, direction);

  // Clean up our resources, Buffers are cleaned up by Legate
  CHECK_CUFFT(cufftDestroy(plan));
}

template <CuNumericFFTType FFT_TYPE, LegateTypeCode CODE_OUT, LegateTypeCode CODE_IN, int32_t DIM>
struct FFTImplBody<VariantKind::GPU, FFT_TYPE, CODE_OUT, CODE_IN, DIM> {
  using INPUT_TYPE  = legate_type_of<CODE_IN>;
  using OUTPUT_TYPE = legate_type_of<CODE_OUT>;

  __host__ void operator()(AccessorWO<OUTPUT_TYPE, DIM> out,
                           AccessorRO<INPUT_TYPE, DIM> in,
                           const Rect<DIM>& out_rect,
                           const Rect<DIM>& in_rect,
                           std::vector<int64_t>& axes,
                           CuNumericFFTDirection direction,
                           bool operate_over_axes) const
  {
    // FFTs are computed as 1D over different axes. Slower than performing the full FFT in a single
    // step
    if (operate_over_axes) {
      // R2C / C2R always only 1D on a single axis (when performed over axes)
      if constexpr (FFT_TYPE != CUNUMERIC_FFT_Z2Z && FFT_TYPE != CUNUMERIC_FFT_C2C) {
        cufft_over_axis_r2c_c2r<DIM, OUTPUT_TYPE, INPUT_TYPE>(
          out, in, out_rect, in_rect, axes, FFT_TYPE, direction);
      }
      // C2C can be multiple 1D dimensions over axes
      else {
        cufft_over_axes_c2c<DIM, OUTPUT_TYPE, INPUT_TYPE>(
          out, in, out_rect, in_rect, axes, FFT_TYPE, direction);
      }
    }
    // If we have one axis per dimension, then it can be done as a single operation (more
    // performant)
    else {
      // FFTs are computed as a single step of DIM
      cufft_operation<DIM, OUTPUT_TYPE, INPUT_TYPE>(
        out, in, out_rect, in_rect, axes, FFT_TYPE, direction);
    }
  }
};

/*static*/ void FFTTask::gpu_variant(TaskContext& context)
{
  fft_template<VariantKind::GPU>(context);
};

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { FFTTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
