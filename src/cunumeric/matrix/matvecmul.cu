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

#include "cunumeric/matrix/matvecmul.h"
#include "cunumeric/matrix/matvecmul_template.inl"

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;

template <>
struct MatVecMulImplBody<VariantKind::GPU, LegateTypeCode::FLOAT_LT> {
  void operator()(size_t m,
                  size_t n,
                  float* lhs,
                  const float* mat,
                  const float* vec,
                  size_t mat_stride,
                  bool transpose_mat)
  {
    cublasHandle_t cublas_handle = Core::get_cublas();
    // Update the stream because the CUDA hijack can't see inside cuBLAS
    cudaStream_t task_stream;
    cudaStreamCreate(&task_stream);
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const float alpha = 1.f;
    const float beta  = 0.f;

    auto trans = transpose_mat ? CUBLAS_OP_N : CUBLAS_OP_T;
    CHECK_CUBLAS(
      cublasSgemv(cublas_handle, trans, n, m, &alpha, mat, mat_stride, vec, 1, &beta, lhs, 1));

    cudaStreamDestroy(task_stream);
  }
};

template <>
struct MatVecMulImplBody<VariantKind::GPU, LegateTypeCode::DOUBLE_LT> {
  void operator()(size_t m,
                  size_t n,
                  double* lhs,
                  const double* mat,
                  const double* vec,
                  size_t mat_stride,
                  bool transpose_mat)
  {
    cublasHandle_t cublas_handle = Core::get_cublas();
    // Update the stream because the CUDA hijack can't see inside cuBLAS
    cudaStream_t task_stream;
    cudaStreamCreate(&task_stream);
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const double alpha = 1.f;
    const double beta  = 0.f;

    auto trans = transpose_mat ? CUBLAS_OP_N : CUBLAS_OP_T;
    CHECK_CUBLAS(
      cublasDgemv(cublas_handle, trans, n, m, &alpha, mat, mat_stride, vec, 1, &beta, lhs, 1));

    cudaStreamDestroy(task_stream);
  }
};

template <>
struct MatVecMulImplBody<VariantKind::GPU, LegateTypeCode::HALF_LT> {
  void operator()(size_t m,
                  size_t n,
                  float* lhs,
                  const __half* mat,
                  const __half* vec,
                  size_t mat_stride,
                  bool transpose_mat)
  {
    cublasHandle_t cublas_handle = Core::get_cublas();
    // Update the stream because the CUDA hijack can't see inside cuBLAS
    cudaStream_t task_stream;
    cudaStreamCreate(&task_stream);
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const float alpha = 1.f;
    const float beta  = 0.f;

    // Use SgemmEx here since there is no half precision gemv yet
    if (transpose_mat) {
      CHECK_CUBLAS(cublasSgemmEx(cublas_handle,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 n,
                                 1,
                                 m,
                                 &alpha,
                                 mat,
                                 CUDA_R_16F,
                                 mat_stride,
                                 vec,
                                 CUDA_R_16F,
                                 m,
                                 &beta,
                                 lhs,
                                 CUDA_R_32F,
                                 n));
    } else {
      CHECK_CUBLAS(cublasSgemmEx(cublas_handle,
                                 CUBLAS_OP_T,
                                 CUBLAS_OP_N,
                                 m,
                                 1,
                                 n,
                                 &alpha,
                                 mat,
                                 CUDA_R_16F,
                                 mat_stride,
                                 vec,
                                 CUDA_R_16F,
                                 n,
                                 &beta,
                                 lhs,
                                 CUDA_R_32F,
                                 m));
    }

    cudaStreamDestroy(task_stream);
  }
};

/*static*/ void MatVecMulTask::gpu_variant(TaskContext& context)
{
  matvecmul_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
