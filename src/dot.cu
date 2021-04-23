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

#include "cuda_help.h"
#include "dot.h"
#include "proj.h"

using namespace Legion;

namespace legate {
namespace numpy {

template <>
/*static*/ void DotTask<__half>::gpu_variant(const Task* task,
                                             const std::vector<PhysicalRegion>& regions,
                                             Context ctx,
                                             Runtime* runtime)
{
  cublasHandle_t cublas_handle = Core::get_cublas();
  // Update the stream because the CUDA hijack can't see inside cuBLAS
  cudaStream_t task_stream;
  cudaStreamCreate(&task_stream);
  CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));
  // Default parameters
  const float alpha = 1.f;
  // Now we can do the operation
  LegateDeserializer derez(task->args, task->arglen);
  const bool partial  = derez.unpack_bool();
  const int extra_dim = derez.unpack_dimension();
  const int dim       = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      // This has to be matrix vector
      const Rect<1> out_rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (out_rect.empty()) return;
      void* out_ptr;
#if 1
      const float beta = (task->regions[0].privilege == READ_WRITE) ? 1.f : 0.f;
      if (partial) {
        if (task->regions[0].privilege == READ_WRITE) {
          const AccessorRW<float, 1> out =
            (extra_dim >= 0)
              ? derez.unpack_accessor_RW<float, 1>(
                  regions[0], out_rect, 1 /*out extra dim*/, task->index_point[extra_dim])
              : derez.unpack_accessor_RW<float, 1>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect);
        } else {
          const AccessorWO<float, 1> out =
            (extra_dim >= 0)
              ? derez.unpack_accessor_WO<float, 1>(
                  regions[0], out_rect, 1 /*out extra dim*/, task->index_point[extra_dim])
              : derez.unpack_accessor_WO<float, 1>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect);
        }
      } else {
        if (task->regions[0].privilege == READ_WRITE) {
          const AccessorRW<__half, 1> out =
            (extra_dim >= 0)
              ? derez.unpack_accessor_RW<__half, 1>(
                  regions[0], out_rect, 1 /*out extra dim*/, task->index_point[extra_dim])
              : derez.unpack_accessor_RW<__half, 1>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect);
        } else {
          const AccessorWO<__half, 1> out =
            (extra_dim >= 0)
              ? derez.unpack_accessor_WO<__half, 1>(
                  regions[0], out_rect, 1 /*out extra dim*/, task->index_point[extra_dim])
              : derez.unpack_accessor_WO<__half, 1>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect);
        }
      }
#else
      const float beta = task->is_index_space ? 1.f : 0.f;
      if (partial) {
        if (task->is_index_space) {
          const AccessorRD<SumReduction<float>, true /*exclusive*/, 1> out =
            derez.unpack_accessor_RD<SumReduction<float>, true, 1>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect);
        } else {
          const AccessorWO<float, 1> out = derez.unpack_accessor_WO<float, 1>(regions[0], out_rect);
          out_ptr                        = out.ptr(out_rect);
        }
      } else {
        if (task->is_index_space) {
          const AccessorRD<SumReduction<__half>, true /*exclusive*/, 1> out =
            derez.unpack_accessor_RD<SumReduction<__half>, true, 1>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect);
        } else {
          const AccessorWO<__half, 1> out =
            derez.unpack_accessor_WO<__half, 1>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect);
        }
      }
#endif
      const int dim1 = derez.unpack_dimension();
      if (dim1 == 1) {
        const Rect<1> in1_rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
        if (in1_rect.empty()) return;
        const AccessorRO<__half, 1> in1 = derez.unpack_accessor_RO<__half, 1>(regions[1], in1_rect);
        const __half* in1_ptr           = in1.ptr(in1_rect);

        const int dim2 = derez.unpack_dimension();
        assert(dim2 == 2);
        const Rect<2> in2_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
        if (in2_rect.empty()) return;
        const AccessorRO<__half, 2> in2 = derez.unpack_accessor_RO<__half, 2>(regions[2], in2_rect);
        // Construct the rect we actually want to do the math for
        const Rect<2> act_rect(Point<2>(in1_rect.lo[0], out_rect.lo[0]),
                               Point<2>(in1_rect.hi[0], out_rect.hi[0]));
        assert(in2_rect.contains(act_rect));
        size_t in2_strides[2];
        const __half* in2_ptr = in2.ptr(act_rect, in2_strides);
        const coord_t m       = (act_rect.hi[0] - act_rect.lo[0]) + 1;
        const coord_t n       = (act_rect.hi[1] - act_rect.lo[1]) + 1;

        // Use SgemmEx here since there is no half precision gemv yet
        CHECK_CUBLAS(cublasSgemmEx(cublas_handle,
                                   CUBLAS_OP_N,
                                   CUBLAS_OP_N,
                                   n,
                                   1,
                                   m,
                                   &alpha,
                                   in2_ptr,
                                   CUDA_R_16F,
                                   in2_strides[0],
                                   in1_ptr,
                                   CUDA_R_16F,
                                   m,
                                   &beta,
                                   out_ptr,
                                   partial ? CUDA_R_32F : CUDA_R_16F,
                                   n));
      } else {
        assert(dim1 == 2);
        const Rect<2> in1_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
        if (in1_rect.empty()) return;
        const AccessorRO<__half, 2> in1 = derez.unpack_accessor_RO<__half, 2>(regions[1], in1_rect);

        const int dim2 = derez.unpack_dimension();
        assert(dim2 == 1);
        const Rect<1> in2_rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
        if (in2_rect.empty()) return;
        const AccessorRO<__half, 1> in2 = derez.unpack_accessor_RO<__half, 1>(regions[2], in2_rect);
        const __half* in2_ptr           = in2.ptr(in2_rect);

        // Construct the rect we actually want to do the math for
        const Rect<2> act_rect(Point<2>(out_rect.lo[0], in2_rect.lo[0]),
                               Point<2>(out_rect.hi[0], in2_rect.hi[0]));
        assert(in1_rect.contains(act_rect));
        size_t in1_strides[2];
        const __half* in1_ptr = in1.ptr(act_rect, in1_strides);
        const coord_t m       = (act_rect.hi[0] - act_rect.lo[0]) + 1;
        const coord_t n       = (act_rect.hi[1] - act_rect.lo[1]) + 1;

        // Use SgemmEx here since there is no half precision gemv yet
        CHECK_CUBLAS(cublasSgemmEx(cublas_handle,
                                   CUBLAS_OP_T,
                                   CUBLAS_OP_N,
                                   m,
                                   1,
                                   n,
                                   &alpha,
                                   in1_ptr,
                                   CUDA_R_16F,
                                   in1_strides[0],
                                   in2_ptr,
                                   CUDA_R_16F,
                                   n,
                                   &beta,
                                   out_ptr,
                                   partial ? CUDA_R_32F : CUDA_R_16F,
                                   m));
      }
      break;
    }
    case 2: {
      // This has to be matrix multiply for us right now
      const float beta       = (task->regions[0].privilege == READ_WRITE) ? 1.f : 0.f;
      const Rect<2> out_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (out_rect.empty()) return;
      void* out_ptr;
      size_t out_strides[2];
      if (partial) {
        // We're doing partial accumulation so the field size is a float
        if (task->regions[0].privilege == READ_WRITE) {
          const AccessorRW<float, 2> out =
            (extra_dim >= 0) ? derez.unpack_accessor_RW<float, 2>(
                                 regions[0], out_rect, extra_dim, task->index_point[extra_dim])
                             : derez.unpack_accessor_RW<float, 2>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect, out_strides);
        } else {
          const AccessorWO<float, 2> out =
            (extra_dim >= 0) ? derez.unpack_accessor_WO<float, 2>(
                                 regions[0], out_rect, extra_dim, task->index_point[extra_dim])
                             : derez.unpack_accessor_WO<float, 2>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect, out_strides);
        }
      } else {
        if (task->regions[0].privilege == READ_WRITE) {
          const AccessorRW<__half, 2> out =
            (extra_dim >= 0) ? derez.unpack_accessor_RW<__half, 2>(
                                 regions[0], out_rect, extra_dim, task->index_point[extra_dim])
                             : derez.unpack_accessor_RW<__half, 2>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect, out_strides);
        } else {
          const AccessorWO<__half, 2> out =
            (extra_dim >= 0) ? derez.unpack_accessor_WO<__half, 2>(
                                 regions[0], out_rect, extra_dim, task->index_point[extra_dim])
                             : derez.unpack_accessor_WO<__half, 2>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect, out_strides);
        }
      }
      const coord_t m = (out_rect.hi[0] - out_rect.lo[0]) + 1;
      const coord_t n = (out_rect.hi[1] - out_rect.lo[1]) + 1;

      const int dim1 = derez.unpack_dimension();
      assert(dim1 == 2);
      const Rect<2> in1_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (in1_rect.empty()) return;
      const AccessorRO<__half, 2> in1 = derez.unpack_accessor_RO<__half, 2>(regions[1], in1_rect);
      size_t in1_strides[2];
      const __half* in1_ptr = in1.ptr(in1_rect, in1_strides);
      assert(m == ((in1_rect.hi[0] - in1_rect.lo[0]) + 1));
      const coord_t k = (in1_rect.hi[1] - in1_rect.lo[1]) + 1;

      const int dim2 = derez.unpack_dimension();
      assert(dim2 == 2);
      const Rect<2> in2_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (in2_rect.empty()) return;
      const AccessorRO<__half, 2> in2 = derez.unpack_accessor_RO<__half, 2>(regions[2], in2_rect);
      size_t in2_strides[2];
      const __half* in2_ptr = in2.ptr(in2_rect, in2_strides);
      assert(k == ((in2_rect.hi[0] - in2_rect.lo[0]) + 1));
      assert(n == ((in2_rect.hi[1] - in2_rect.lo[1]) + 1));

      // cublas is dumb and doesn't support row-major, so reverse the matrix
      // order to help cublas think things are column-major
      // effectively we get NxM = NxK * KxM
      // Use the extended sgemm interface so we can use tensor cores
      // if they are available for this matrix shape and GPU
      CHECK_CUBLAS(cublasSgemmEx(cublas_handle,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 n,
                                 m,
                                 k,
                                 &alpha,
                                 in2_ptr,
                                 CUDA_R_16F,
                                 in2_strides[0],
                                 in1_ptr,
                                 CUDA_R_16F,
                                 in1_strides[0],
                                 &beta,
                                 out_ptr,
                                 partial ? CUDA_R_32F : CUDA_R_16F,
                                 out_strides[0]));
      break;
    }
    default: assert(false);  // we don't support any other updates
  }
}

template <>
/*static*/ void DotTask<float>::gpu_variant(const Task* task,
                                            const std::vector<PhysicalRegion>& regions,
                                            Context ctx,
                                            Runtime* runtime)
{
  cublasHandle_t cublas_handle = Core::get_cublas();
  // Update the stream because the CUDA hijack can't see inside cuBLAS
  cudaStream_t task_stream;
  cudaStreamCreate(&task_stream);
  CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));
  // Default parameters
  const float alpha = 1.f;
  // Now we can do the operation
  LegateDeserializer derez(task->args, task->arglen);
  const int extra_dim = derez.unpack_dimension();
  const int dim       = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      // This has to be matrix vector
      const Rect<1> out_rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (out_rect.empty()) return;
      float* out_ptr;
#if 1
      const float beta = (task->regions[0].privilege == READ_WRITE) ? 1.f : 0.f;
      if (task->regions[0].privilege == READ_WRITE) {
        const AccessorRW<float, 1> out =
          (extra_dim >= 0)
            ? derez.unpack_accessor_RW<float, 1>(
                regions[0], out_rect, 1 /*out extram dim*/, task->index_point[extra_dim])
            : derez.unpack_accessor_RW<float, 1>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect);
      } else {
        const AccessorWO<float, 1> out =
          (extra_dim >= 0)
            ? derez.unpack_accessor_WO<float, 1>(
                regions[0], out_rect, 1 /*out extra dim*/, task->index_point[extra_dim])
            : derez.unpack_accessor_WO<float, 1>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect);
      }
#else
      const float beta = task->is_index_space ? 1.f : 0.f;
      if (task->is_index_space) {
        const AccessorRD<SumReduction<float>, true /*exclusive*/, 1> out =
          derez.unpack_accessor_RD<SumReduction<float>, true, 1>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect);
      } else {
        const AccessorWO<float, 1> out = derez.unpack_accessor_WO<float, 1>(regions[0], out_rect);
        out_ptr                        = out.ptr(out_rect);
      }
#endif
      const int dim1 = derez.unpack_dimension();
      if (dim1 == 1) {
        const Rect<1> in1_rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
        if (in1_rect.empty()) return;
        const AccessorRO<float, 1> in1 = derez.unpack_accessor_RO<float, 1>(regions[1], in1_rect);
        const float* in1_ptr           = in1.ptr(in1_rect);

        const int dim2 = derez.unpack_dimension();
        assert(dim2 == 2);
        const Rect<2> in2_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
        if (in2_rect.empty()) return;
        const AccessorRO<float, 2> in2 = derez.unpack_accessor_RO<float, 2>(regions[2], in2_rect);
        // Construct the rect we actually want to do the math for
        const Rect<2> act_rect(Point<2>(in1_rect.lo[0], out_rect.lo[0]),
                               Point<2>(in1_rect.hi[0], out_rect.hi[0]));
        assert(in2_rect.contains(act_rect));
        size_t in2_strides[2];
        const float* in2_ptr = in2.ptr(act_rect, in2_strides);
        const coord_t m      = (act_rect.hi[0] - act_rect.lo[0]) + 1;
        const coord_t n      = (act_rect.hi[1] - act_rect.lo[1]) + 1;

        CHECK_CUBLAS(cublasSgemv(cublas_handle,
                                 CUBLAS_OP_N,
                                 n,
                                 m,
                                 &alpha,
                                 in2_ptr,
                                 in2_strides[0],
                                 in1_ptr,
                                 1,
                                 &beta,
                                 out_ptr,
                                 1));
      } else {
        assert(dim1 == 2);
        const Rect<2> in1_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
        if (in1_rect.empty()) return;
        const AccessorRO<float, 2> in1 = derez.unpack_accessor_RO<float, 2>(regions[1], in1_rect);

        const int dim2 = derez.unpack_dimension();
        assert(dim2 == 1);
        const Rect<1> in2_rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
        if (in2_rect.empty()) return;
        const AccessorRO<float, 1> in2 = derez.unpack_accessor_RO<float, 1>(regions[2], in2_rect);
        const float* in2_ptr           = in2.ptr(in2_rect);

        // Construct the rect we actually want to do the math for
        const Rect<2> act_rect(Point<2>(out_rect.lo[0], in2_rect.lo[0]),
                               Point<2>(out_rect.hi[0], in2_rect.hi[0]));
        assert(in1_rect.contains(act_rect));
        size_t in1_strides[2];
        const float* in1_ptr = in1.ptr(act_rect, in1_strides);
        const coord_t m      = (act_rect.hi[0] - act_rect.lo[0]) + 1;
        const coord_t n      = (act_rect.hi[1] - act_rect.lo[1]) + 1;

        CHECK_CUBLAS(cublasSgemv(cublas_handle,
                                 CUBLAS_OP_T,
                                 n,
                                 m,
                                 &alpha,
                                 in1_ptr,
                                 in1_strides[0],
                                 in2_ptr,
                                 1,
                                 &beta,
                                 out_ptr,
                                 1));
      }
      break;
    }
    case 2: {
      // This has to be matrix multiply for us right now
      const float beta       = (task->regions[0].privilege == READ_WRITE) ? 1.f : 0.f;
      const Rect<2> out_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (out_rect.empty()) return;
      float* out_ptr;
      size_t out_strides[2];
      if (task->regions[0].privilege == READ_WRITE) {
        const AccessorRW<float, 2> out =
          (extra_dim >= 0) ? derez.unpack_accessor_RW<float, 2>(
                               regions[0], out_rect, extra_dim, task->index_point[extra_dim])
                           : derez.unpack_accessor_RW<float, 2>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect, out_strides);
      } else {
        const AccessorWO<float, 2> out =
          (extra_dim >= 0) ? derez.unpack_accessor_WO<float, 2>(
                               regions[0], out_rect, extra_dim, task->index_point[extra_dim])
                           : derez.unpack_accessor_WO<float, 2>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect, out_strides);
      }
      const coord_t m = (out_rect.hi[0] - out_rect.lo[0]) + 1;
      const coord_t n = (out_rect.hi[1] - out_rect.lo[1]) + 1;

      const int dim1 = derez.unpack_dimension();
      assert(dim1 == 2);
      const Rect<2> in1_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (in1_rect.empty()) return;
      const AccessorRO<float, 2> in1 = derez.unpack_accessor_RO<float, 2>(regions[1], in1_rect);
      size_t in1_strides[2];
      const float* in1_ptr = in1.ptr(in1_rect, in1_strides);
      assert(m == ((in1_rect.hi[0] - in1_rect.lo[0]) + 1));
      const coord_t k = (in1_rect.hi[1] - in1_rect.lo[1]) + 1;

      const int dim2 = derez.unpack_dimension();
      assert(dim2 == 2);
      const Rect<2> in2_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (in2_rect.empty()) return;
      const AccessorRO<float, 2> in2 = derez.unpack_accessor_RO<float, 2>(regions[2], in2_rect);
      size_t in2_strides[2];
      const float* in2_ptr = in2.ptr(in2_rect, in2_strides);
      assert(k == ((in2_rect.hi[0] - in2_rect.lo[0]) + 1));
      assert(n == ((in2_rect.hi[1] - in2_rect.lo[1]) + 1));

      // cublas is dumb and doesn't support row-major, so reverse the matrix
      // order to help cublas think things are column-major
      // effectively we get NxM = NxK * KxM
      // Use the extended sgemm interface so we can use tensor cores
      // if they are available for this matrix shape and GPU
      CHECK_CUBLAS(cublasSgemmEx(cublas_handle,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 n,
                                 m,
                                 k,
                                 &alpha,
                                 in2_ptr,
                                 CUDA_R_32F,
                                 in2_strides[0],
                                 in1_ptr,
                                 CUDA_R_32F,
                                 in1_strides[0],
                                 &beta,
                                 out_ptr,
                                 CUDA_R_32F,
                                 out_strides[0]));
      break;
    }
    default: assert(false);  // we don't support any other updates
  }
}

template <>
/*static*/ void DotTask<double>::gpu_variant(const Task* task,
                                             const std::vector<PhysicalRegion>& regions,
                                             Context ctx,
                                             Runtime* runtime)
{
  cublasHandle_t cublas_handle = Core::get_cublas();
  // Update the stream because the CUDA hijack can't see inside cuBLAS
  cudaStream_t task_stream;
  cudaStreamCreate(&task_stream);
  CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));
  // Default parameters
  const double alpha = 1.0;
  // Now we can do the operation
  LegateDeserializer derez(task->args, task->arglen);
  const int extra_dim = derez.unpack_dimension();
  const int dim       = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      // This has to be matrix vector
      const Rect<1> out_rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (out_rect.empty()) return;
      double* out_ptr;
#if 1
      const double beta = (task->regions[0].privilege == READ_WRITE) ? 1.0 : 0.0;
      if (task->regions[0].privilege == READ_WRITE) {
        const AccessorRW<double, 1> out =
          (extra_dim >= 0)
            ? derez.unpack_accessor_RW<double, 1>(
                regions[0], out_rect, 1 /*out extra dim*/, task->index_point[extra_dim])
            : derez.unpack_accessor_RW<double, 1>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect);
      } else {
        const AccessorWO<double, 1> out =
          (extra_dim >= 0)
            ? derez.unpack_accessor_WO<double, 1>(
                regions[0], out_rect, 1 /*out extra dim*/, task->index_point[extra_dim])
            : derez.unpack_accessor_WO<double, 1>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect);
      }
#else
      const double beta = task->is_index_space ? 1.0 : 0.0;
      if (task->is_index_space) {
        const AccessorRD<SumReduction<double>, true /*exclusive*/, 1> out =
          derez.unpack_accessor_RD<SumReduction<double>, true, 1>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect);
      } else {
        const AccessorWO<double, 1> out = derez.unpack_accessor_WO<double, 1>(regions[0], out_rect);
        out_ptr                         = out.ptr(out_rect);
      }
#endif
      const int dim1 = derez.unpack_dimension();
      if (dim1 == 1) {
        const Rect<1> in1_rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
        if (in1_rect.empty()) return;
        const AccessorRO<double, 1> in1 = derez.unpack_accessor_RO<double, 1>(regions[1], in1_rect);
        const double* in1_ptr           = in1.ptr(in1_rect);

        const int dim2 = derez.unpack_dimension();
        assert(dim2 == 2);
        const Rect<2> in2_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
        if (in2_rect.empty()) return;
        const AccessorRO<double, 2> in2 = derez.unpack_accessor_RO<double, 2>(regions[2], in2_rect);
        // Construct the rect we actually want to do the math for
        const Rect<2> act_rect(Point<2>(in1_rect.lo[0], out_rect.lo[0]),
                               Point<2>(in1_rect.hi[0], out_rect.hi[0]));
        assert(in2_rect.contains(act_rect));
        size_t in2_strides[2];
        const double* in2_ptr = in2.ptr(act_rect, in2_strides);
        const coord_t m       = (act_rect.hi[0] - act_rect.lo[0]) + 1;
        const coord_t n       = (act_rect.hi[1] - act_rect.lo[1]) + 1;

        CHECK_CUBLAS(cublasDgemv(cublas_handle,
                                 CUBLAS_OP_N,
                                 n,
                                 m,
                                 &alpha,
                                 in2_ptr,
                                 in2_strides[0],
                                 in1_ptr,
                                 1,
                                 &beta,
                                 out_ptr,
                                 1));
      } else {
        assert(dim1 == 2);
        const Rect<2> in1_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
        if (in1_rect.empty()) return;
        const AccessorRO<double, 2> in1 = derez.unpack_accessor_RO<double, 2>(regions[1], in1_rect);

        const int dim2 = derez.unpack_dimension();
        assert(dim2 == 1);
        const Rect<1> in2_rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
        if (in2_rect.empty()) return;
        const AccessorRO<double, 1> in2 = derez.unpack_accessor_RO<double, 1>(regions[2], in2_rect);
        const double* in2_ptr           = in2.ptr(in2_rect);

        // Construct the rect we actually want to do the math for
        const Rect<2> act_rect(Point<2>(out_rect.lo[0], in2_rect.lo[0]),
                               Point<2>(out_rect.hi[0], in2_rect.hi[0]));
        assert(in1_rect.contains(act_rect));
        size_t in1_strides[2];
        const double* in1_ptr = in1.ptr(act_rect, in1_strides);
        const coord_t m       = (act_rect.hi[0] - act_rect.lo[0]) + 1;
        const coord_t n       = (act_rect.hi[1] - act_rect.lo[1]) + 1;

        CHECK_CUBLAS(cublasDgemv(cublas_handle,
                                 CUBLAS_OP_T,
                                 n,
                                 m,
                                 &alpha,
                                 in1_ptr,
                                 in1_strides[0],
                                 in2_ptr,
                                 1,
                                 &beta,
                                 out_ptr,
                                 1));
      }
      break;
    }
    case 2: {
      // This has to be matrix multiply for us right now
      const double beta      = (task->regions[0].privilege == READ_WRITE) ? 1.0 : 0.0;
      const Rect<2> out_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (out_rect.empty()) return;
      double* out_ptr;
      size_t out_strides[2];
      if (task->regions[0].privilege == READ_WRITE) {
        const AccessorRW<double, 2> out =
          (extra_dim >= 0) ? derez.unpack_accessor_RW<double, 2>(
                               regions[0], out_rect, extra_dim, task->index_point[extra_dim])
                           : derez.unpack_accessor_RW<double, 2>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect, out_strides);
      } else {
        const AccessorWO<double, 2> out =
          (extra_dim >= 0) ? derez.unpack_accessor_WO<double, 2>(
                               regions[0], out_rect, extra_dim, task->index_point[extra_dim])
                           : derez.unpack_accessor_WO<double, 2>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect, out_strides);
      }
      const coord_t m = (out_rect.hi[0] - out_rect.lo[0]) + 1;
      const coord_t n = (out_rect.hi[1] - out_rect.lo[1]) + 1;

      const int dim1 = derez.unpack_dimension();
      assert(dim1 == 2);
      const Rect<2> in1_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (in1_rect.empty()) return;
      const AccessorRO<double, 2> in1 = derez.unpack_accessor_RO<double, 2>(regions[1], in1_rect);
      size_t in1_strides[2];
      const double* in1_ptr = in1.ptr(in1_rect, in1_strides);
      assert(m == ((in1_rect.hi[0] - in1_rect.lo[0]) + 1));
      const coord_t k = (in1_rect.hi[1] - in1_rect.lo[1]) + 1;

      const int dim2 = derez.unpack_dimension();
      assert(dim2 == 2);
      const Rect<2> in2_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (in2_rect.empty()) return;
      const AccessorRO<double, 2> in2 = derez.unpack_accessor_RO<double, 2>(regions[2], in2_rect);
      size_t in2_strides[2];
      const double* in2_ptr = in2.ptr(in2_rect, in2_strides);
      assert(k == ((in2_rect.hi[0] - in2_rect.lo[0]) + 1));
      assert(n == ((in2_rect.hi[1] - in2_rect.lo[1]) + 1));

      // cublas is dumb and doesn't support row-major, so reverse the matrix
      // order to help cublas think things are column-major
      // effectively we get NxM = NxK * KxM
      CHECK_CUBLAS(cublasDgemm(cublas_handle,
                               CUBLAS_OP_N,
                               CUBLAS_OP_N,
                               n,
                               m,
                               k,
                               &alpha,
                               in2_ptr,
                               in2_strides[0],
                               in1_ptr,
                               in1_strides[0],
                               &beta,
                               out_ptr,
                               out_strides[0]));
      break;
    }
    default: assert(false);  // we don't support any other updates
  }
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_dot_reduce(const DeferredBuffer<T, 1> buffer,
                    const AccessorRO<T, 1> in1,
                    const AccessorRO<T, 1> in2,
                    const Point<1> origin,
                    const size_t max,
                    const T identity)
{
  T value             = identity;
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < max) {
    const coord_t x = origin[0] + offset;
    value           = in1[x];
    ProdReduction<T>::template fold<true /*exclusive*/>(value, in2[x]);
  }
  fold_output(buffer, value, SumReduction<T>{});
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) legate_buffer_dot_reduce(
  const DeferredBuffer<T, 1> in, const DeferredBuffer<T, 1> out, const size_t max, const T identity)
{
  T value             = identity;
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < max) value = in.read(offset);
  fold_output(out, value, SumReduction<T>{});
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  legate_final_dot_reduce(const DeferredBuffer<T, 1> in,
                          const DeferredReduction<SumReduction<T>> out,
                          const size_t max,
                          const T identity)
{
  T value             = identity;
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < max) value = in.read(offset);
  reduce_output(out, value);
}

template <typename T>
/*static*/ DeferredReduction<SumReduction<T>> DotReducTask<T>::gpu_variant(
  const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int dim = derez.unpack_dimension();
  assert(dim == 1);
  const Rect<1> rect         = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  const AccessorRO<T, 1> in1 = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
  const AccessorRO<T, 1> in2 = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
  DeferredBuffer<T, 1> bufferA;
  size_t volume = 0, blocks = 0;
  if (!rect.empty()) {
    volume = rect.volume();
    blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    Rect<1> bounds(Point<1>(0), Point<1>(blocks - 1));
    bufferA = DeferredBuffer<T, 1>(Memory::GPU_FB_MEM, Domain(bounds));
    legate_dot_reduce<T><<<blocks, THREADS_PER_BLOCK>>>(
      bufferA, in1, in2, rect.lo, volume, SumReduction<T>::identity);
    volume = blocks;
  }
  // Continue reducing buffers until we get down to one small enough that
  // it can be handled by a single CTA and then we can do the final launch
  DeferredBuffer<T, 1> last = bufferA;
  if (volume > THREADS_PER_BLOCK) {
    DeferredBuffer<T, 1> bufferB;
    bool b_initialized = false;
    bool forward       = true;
    while (volume > THREADS_PER_BLOCK) {
      blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      if (!b_initialized) {
        Rect<1> bounds = Rect<1>(Point<1>(0), Point<1>(blocks - 1));
        bufferB        = DeferredBuffer<T, 1>(Memory::GPU_FB_MEM, Domain(bounds));
        b_initialized  = true;
      }
      if (forward) {
        legate_buffer_dot_reduce<T>
          <<<blocks, THREADS_PER_BLOCK>>>(bufferA, bufferB, volume, SumReduction<T>::identity);
        forward = false;
      } else {
        legate_buffer_dot_reduce<T>
          <<<blocks, THREADS_PER_BLOCK>>>(bufferB, bufferA, volume, SumReduction<T>::identity);
        forward = true;
      }
      volume = blocks;
    }
    if (!forward) last = bufferB;
  }
  DeferredReduction<SumReduction<T>> result;
  // One last kernel launch to do the final reduction to a single value
  if (volume > 0)
    legate_final_dot_reduce<T>
      <<<1, THREADS_PER_BLOCK>>>(last, result, volume, SumReduction<T>::identity);
  return result;
}

INSTANTIATE_DEFERRED_REDUCTION_TASK_VARIANT(DotReducTask, SumReduction, gpu_variant)

}  // namespace numpy
}  // namespace legate
