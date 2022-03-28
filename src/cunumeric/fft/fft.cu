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
#include <csignal>

#include "cunumeric/fft/fft.h"
#include "cunumeric/fft/fft_template.inl"

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

using dim_t = long long int;

template <int DIM>
__host__ static inline void cufft_operation(void* output,
                                            void* input,
                                            const Rect<DIM>& out_rect,
                                            const Rect<DIM>& in_rect,
                                            std::vector<int64_t>& axes,
                                            fftType type,
                                            fftDirection direction)
{
    auto stream = get_cached_stream();

    size_t workarea_size = 0;
    size_t num_elements;
    dim_t n[DIM];
    dim_t inembed[DIM];
    dim_t onembed[DIM];

    const Point<DIM> zero   = Point<DIM>::ZEROES();
    const Point<DIM> one    = Point<DIM>::ONES();
    Point<DIM> fft_size_in  =  in_rect.hi -  in_rect.lo + one;
    Point<DIM> fft_size_out = out_rect.hi - out_rect.lo + one;
    num_elements = 1;
    for(int i = 0; i < DIM; ++i) {
      n[i]          = (type == fftType::FFT_R2C || type == fftType::FFT_D2Z) ? fft_size_in[i] : fft_size_out[i];
      inembed[i]    = fft_size_in[i];
      onembed[i]    = fft_size_out[i];
      num_elements *= n[i];
    }

    // Create the plan
    cufftHandle plan;
    CHECK_CUFFT(cufftCreate(&plan));
    CHECK_CUFFT(cufftSetAutoAllocation(plan, 0 /*we'll do the allocation*/));
    CHECK_CUFFT(cufftSetStream(plan, stream));

    // Create the plan and allocate a temporary buffer for it if it needs one
    CHECK_CUFFT(cufftMakePlanMany64(plan, DIM, n, inembed, 1, 1, onembed, 1, 1, (cufftType)type, 1, &workarea_size));

    DeferredBuffer<uint8_t, 1> workarea_buffer;
    if(workarea_size > 0) {
      const Point<1> zero1d(0);
      workarea_buffer =
        DeferredBuffer<uint8_t, 1>(Rect<1>(zero1d, Point<1>(workarea_size - 1)),
                                   Memory::GPU_FB_MEM,
                                   nullptr /*initial*/,
                                   128 /*alignment*/);
      void* workarea = workarea_buffer.ptr(zero1d);
      CHECK_CUFFT(cufftSetWorkArea(plan, workarea));
    }

    // FFT the input data
    CHECK_CUFFT(cufftXtExec(plan, input, output, (int)direction));

    // Clean up our resources, DeferredBuffers are cleaned up by Legion
    CHECK_CUFFT(cufftDestroy(plan));
}

template<int DIM, typename OUTPUT_TYPE, typename INPUT_TYPE>
struct cufft_axes_plan{
  __host__ static inline void execute(cufftHandle plan,
                                      // AccessorWO<OUTPUT_TYPE, DIM> out,
                                      // AccessorRO<INPUT_TYPE, DIM> in,
                                      DeferredBuffer<OUTPUT_TYPE, DIM>& out,
                                      DeferredBuffer<INPUT_TYPE, DIM>& in,
                                      const Rect<DIM>& out_rect,
                                      const Rect<DIM>& in_rect,
                                      int axis,
                                      fftDirection direction) {
      const Point<DIM> zero   = Point<DIM>::ZEROES();
      CHECK_CUFFT(cufftXtExec(plan, (void*)in.ptr(zero), (void*)out.ptr(zero), (int)direction));
  }
};

// For dimensions higher than 2D, we need to iterate through the input volume as 2D slices due to
// limitations of cuFFT indexing in 1D
template<typename OUTPUT_TYPE, typename INPUT_TYPE>
struct cufft_axes_plan<3, OUTPUT_TYPE, INPUT_TYPE>{
  __host__ static inline void execute(cufftHandle plan,
                                      // AccessorWO<OUTPUT_TYPE, 3> out,
                                      // AccessorRO<INPUT_TYPE,  3> in,
                                      DeferredBuffer<OUTPUT_TYPE, 3>& out,
                                      DeferredBuffer<INPUT_TYPE, 3>& in,
                                      const Rect<3>& out_rect,
                                      const Rect<3>& in_rect,
                                      int axis,
                                      fftDirection direction) {
    bool is_inner_axis = (axis == 1);
    if(is_inner_axis) {
      // TODO: use PointInRectIterator<DIM>
      auto num_slices = in_rect.hi[0] - in_rect.lo[0] + 1;
      for(unsigned n = 0; n < num_slices; ++n){
        const Point<3> offset = Point<3>(n, 0, 0);
        CHECK_CUFFT(cufftXtExec(plan, (void*)in.ptr(offset), (void*)out.ptr(offset), (int)direction));
      }
    }
    else {
      const Point<3> zero   = Point<3>::ZEROES();
      CHECK_CUFFT(cufftXtExec(plan, (void*)in.ptr(zero), (void*)out.ptr(zero), (int)direction));
    }
  }
};

// Perform the FFT operation as multiple 1D FFTs along the specified axes (Complex-to-complex case).
// For now, it only supports up to 3D FFTs, but final plan is having support for
// N-dimensional FFTs using this approach.
// See cufft_over_axis_r2c_c2r for the equivalent on a single R2C/C2R axis.
template <int DIM, typename OUTPUT_TYPE, typename INPUT_TYPE>
__host__ static inline void cufft_over_axes_c2c(AccessorWO<OUTPUT_TYPE, DIM> out,
                                                AccessorRO<INPUT_TYPE, DIM> in,
                                                const Rect<DIM>& out_rect,
                                                const Rect<DIM>& in_rect,
                                                std::vector<int64_t>& axes,
                                                fftType type,
                                                fftDirection direction)
{
    auto stream = get_cached_stream();

    size_t workarea_size = 0;
    dim_t n[DIM];
    dim_t inembed[DIM];
    dim_t onembed[DIM];

    // Full volume dimensions / strides
    const Point<DIM> zero   = Point<DIM>::ZEROES();
    const Point<DIM> one    = Point<DIM>::ONES();
    Point<DIM> fft_size_in  =  in_rect.hi -  in_rect.lo + one;
    Point<DIM> fft_size_out = out_rect.hi - out_rect.lo + one;
    size_t num_elements_in  = 1;
    size_t num_elements_out = 1;
    for(int i = 0; i < DIM; ++i) {
      n[i]          = fft_size_out[i];
      inembed[i]    = fft_size_in[i];
      onembed[i]    = fft_size_out[i];
      num_elements_in  *= fft_size_in[i];
      num_elements_out *= fft_size_out[i];
    }

    // Copy input to temporary buffer to perform FFTs one by one
    DeferredBuffer<INPUT_TYPE, DIM> input_buffer(Rect<DIM>(zero, fft_size_in - one),
                                                 Memory::GPU_FB_MEM,
                                                 nullptr /*initial*/,
                                                 128 /*alignment*/);
    CHECK_CUDA(cudaMemcpyAsync(input_buffer.ptr(zero), in.ptr(zero), num_elements_in*sizeof(INPUT_TYPE), cudaMemcpyDefault, stream));

    for(auto& ax : axes) {
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
      for(int i = ax+1; i < DIM; ++i) {
        istride *= fft_size_in[i];
        ostride *= fft_size_out[i];
      }
      dim_t idist = (ax == DIM-1) ? fft_size_in[ax] : 1;
      dim_t odist = (ax == DIM-1) ? fft_size_out[ax] : 1;

      // Create the plan and allocate a temporary buffer for it if it needs one
      CHECK_CUFFT(cufftMakePlanMany64(plan, 1, &size_1d, inembed, istride, idist, onembed, ostride, odist, (cufftType)type, batches, &workarea_size));

      DeferredBuffer<uint8_t, 1> workarea_buffer;
      if(workarea_size > 0) {
        const Point<1> zero1d(0);
        workarea_buffer =
          DeferredBuffer<uint8_t, 1>(Rect<1>(zero1d, Point<1>(workarea_size - 1)),
                                     Memory::GPU_FB_MEM,
                                     nullptr /*initial*/,
                                     128 /*alignment*/);
        void* workarea = workarea_buffer.ptr(zero1d);
        CHECK_CUFFT(cufftSetWorkArea(plan, workarea));
      }

      // TODO: following function only correct for DIM <= 3. Fix for N-DIM case
      cufft_axes_plan<DIM, INPUT_TYPE, INPUT_TYPE>::execute(plan, input_buffer, input_buffer, out_rect, in_rect, ax, direction);

      // Clean up our resources, DeferredBuffers are cleaned up by Legion
      CHECK_CUFFT(cufftDestroy(plan));
    }
    CHECK_CUDA(cudaMemcpyAsync(out.ptr(zero), input_buffer.ptr(zero), num_elements_out*sizeof(OUTPUT_TYPE), cudaMemcpyDefault, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

// Perform the FFT operation as multiple 1D FFTs along the specified axes, single R2C/C2R operation.
template <int DIM, typename OUTPUT_TYPE, typename INPUT_TYPE>
__host__ static inline void cufft_over_axis_r2c_c2r(AccessorWO<OUTPUT_TYPE, DIM> out,
                                                    AccessorRO<INPUT_TYPE, DIM> in,
                                                    const Rect<DIM>& out_rect,
                                                    const Rect<DIM>& in_rect,
                                                    std::vector<int64_t>& axes,
                                                    fftType type,
                                                    fftDirection direction)
{
    auto stream = get_cached_stream();

    size_t workarea_size = 0;
    dim_t n[DIM];
    dim_t inembed[DIM];
    dim_t onembed[DIM];

    // Full volume dimensions / strides
    const Point<DIM> zero   = Point<DIM>::ZEROES();
    const Point<DIM> one    = Point<DIM>::ONES();
    Point<DIM> fft_size_in  =  in_rect.hi -  in_rect.lo + one;
    Point<DIM> fft_size_out = out_rect.hi - out_rect.lo + one;
    size_t num_elements_in  = 1;
    size_t num_elements_out = 1;
    for(int i = 0; i < DIM; ++i) {
      n[i]          = (direction == fftDirection::FFT_FORWARD) ? fft_size_in[i] : fft_size_out[i];
      inembed[i]    = fft_size_in[i];
      onembed[i]    = fft_size_out[i];
      num_elements_in  *= fft_size_in[i];
      num_elements_out *= fft_size_out[i];
    }

    // Copy input to temporary buffer to perform FFTs one by one
    DeferredBuffer<INPUT_TYPE, DIM> input_buffer(Rect<DIM>(zero, fft_size_in - one),
                                                 Memory::GPU_FB_MEM,
                                                 nullptr /*initial*/,
                                                 128 /*alignment*/);
    CHECK_CUDA(cudaMemcpyAsync(input_buffer.ptr(zero), in.ptr(zero), num_elements_in*sizeof(INPUT_TYPE), cudaMemcpyDefault, stream));

    DeferredBuffer<OUTPUT_TYPE, DIM> output_buffer(Rect<DIM>(zero, fft_size_out - one),
                                                  Memory::GPU_FB_MEM,
                                                  nullptr /*initial*/,
                                                  128 /*alignment*/);
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
    dim_t batches = (direction == fftDirection::FFT_FORWARD) ? num_elements_in : num_elements_out;
                  batches = (DIM == 3 && axis == 1) ? n[2] : batches / n[axis];
    dim_t istride = 1;
    dim_t ostride = 1;
    for(int i = axis+1; i < DIM; ++i) {
      istride *= fft_size_in[i];
      ostride *= fft_size_out[i];
    }
    dim_t idist = (axis == DIM-1) ? fft_size_in[axis]  : 1;
    dim_t odist = (axis == DIM-1) ? fft_size_out[axis] : 1;

    // Create the plan and allocate a temporary buffer for it if it needs one
    CHECK_CUFFT(cufftMakePlanMany64(plan, 1, &size_1d, inembed, istride, idist, onembed, ostride, odist, (cufftType)type, batches, &workarea_size));

    DeferredBuffer<uint8_t, 1> workarea_buffer;
    if(workarea_size > 0) {
      const Point<1> zero1d(0);
      workarea_buffer =
        DeferredBuffer<uint8_t, 1>(Rect<1>(zero1d, Point<1>(workarea_size - 1)),
                                   Memory::GPU_FB_MEM,
                                   nullptr /*initial*/,
                                   128 /*alignment*/);
      void* workarea = workarea_buffer.ptr(zero1d);
      CHECK_CUFFT(cufftSetWorkArea(plan, workarea));
    }

    cufft_axes_plan<DIM, OUTPUT_TYPE, INPUT_TYPE>::execute(plan, output_buffer, input_buffer, out_rect, in_rect, axis, direction);

    CHECK_CUDA(cudaMemcpyAsync(out.ptr(zero), output_buffer.ptr(zero), num_elements_out*sizeof(OUTPUT_TYPE), cudaMemcpyDefault, stream));

    // Clean up our resources, DeferredBuffers are cleaned up by Legion
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <fftType FFT_TYPE, LegateTypeCode CODE_OUT, LegateTypeCode CODE_IN, int DIM>
struct FFTImplBody<VariantKind::GPU, FFT_TYPE, CODE_OUT, CODE_IN, DIM> {
  using INPUT_TYPE  = legate_type_of<CODE_IN>;
  using OUTPUT_TYPE = legate_type_of<CODE_OUT>;

  __host__ void operator()(AccessorWO<OUTPUT_TYPE, DIM> out,
                           AccessorRO<INPUT_TYPE, DIM> in,
                           const Rect<DIM>& out_rect,
                           const Rect<DIM>& in_rect,
                           std::vector<int64_t>& axes,
                           fftDirection direction,
                           bool operate_over_axes) const
  {
    const Point<DIM> zero = Point<DIM>::ZEROES();
    void* out_ptr = (void*) out.ptr(zero);
    void* in_ptr  = (void*) in.ptr(zero);

    // FFTs are computed as 1D over different axes. Slower than performing the full FFT in a single step
    if(operate_over_axes) {
      // R2C / C2R always only 1D on a single axis (when performed over axes)
      if(FFT_TYPE != fftType::FFT_Z2Z && FFT_TYPE != fftType::FFT_C2C) {
        cufft_over_axis_r2c_c2r<DIM, OUTPUT_TYPE, INPUT_TYPE>(out, in, out_rect, in_rect, axes, FFT_TYPE, direction);
      }
      // C2C can be multiple 1D dimensions over axes
      else {
        cufft_over_axes_c2c<DIM, OUTPUT_TYPE, INPUT_TYPE>(out, in, out_rect, in_rect, axes, FFT_TYPE, direction);        
      }
    }
    // If we have one axis per dimension, then it can be done as a single operation (more performant)
    else {
      // FFTs are computed as a single step of DIM
      cufft_operation<DIM>(out_ptr, in_ptr, out_rect, in_rect, axes, FFT_TYPE, direction);      
    }
  }
};

/*static*/ void FFTTask::gpu_variant(TaskContext& context)
{
  fft_template<VariantKind::GPU>(context);
};

}  // namespace cunumeric
