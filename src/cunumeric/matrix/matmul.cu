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

#include "cunumeric/matrix/matmul.h"
#include "cunumeric/matrix/matmul_template.inl"

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;

template <>
struct MatMulImplBody<VariantKind::GPU, LegateTypeCode::FLOAT_LT> {
  void operator()(size_t m,
                  size_t n,
                  size_t k,
                  float* lhs,
                  const float* rhs1,
                  const float* rhs2,
                  size_t lhs_stride,
                  size_t rhs1_stride,
                  size_t rhs2_stride,
                  bool rhs1_transposed,
                  bool rhs2_transposed)
  {
    cublasHandle_t cublas_handle = Core::get_cublas();
    // Update the stream because the CUDA hijack can't see inside cuBLAS
    cudaStream_t task_stream;
    cudaStreamCreate(&task_stream);
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const float alpha = 1.f;
    const float beta  = 0.f;

    // cublas is dumb and doesn't support row-major, so reverse the matrix
    // order to help cublas think things are column-major
    // effectively we get NxM = NxK * KxM
    // Use the extended sgemm interface so we can use tensor cores
    // if they are available for this matrix shape and GPU
    CHECK_CUBLAS(cublasSgemmEx(cublas_handle,
                               rhs2_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                               rhs1_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                               n,
                               m,
                               k,
                               &alpha,
                               rhs2,
                               CUDA_R_32F,
                               rhs2_stride,
                               rhs1,
                               CUDA_R_32F,
                               rhs1_stride,
                               &beta,
                               lhs,
                               CUDA_R_32F,
                               lhs_stride));

    cudaStreamDestroy(task_stream);
  }
};

template <>
struct MatMulImplBody<VariantKind::GPU, LegateTypeCode::DOUBLE_LT> {
  void operator()(size_t m,
                  size_t n,
                  size_t k,
                  double* lhs,
                  const double* rhs1,
                  const double* rhs2,
                  size_t lhs_stride,
                  size_t rhs1_stride,
                  size_t rhs2_stride,
                  bool rhs1_transposed,
                  bool rhs2_transposed)
  {
    cublasHandle_t cublas_handle = Core::get_cublas();
    // Update the stream because the CUDA hijack can't see inside cuBLAS
    cudaStream_t task_stream;
    cudaStreamCreate(&task_stream);
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const double alpha = 1.f;
    const double beta  = 0.f;

    // cublas is dumb and doesn't support row-major, so reverse the matrix
    // order to help cublas think things are column-major
    // effectively we get NxM = NxK * KxM
    CHECK_CUBLAS(cublasDgemm(cublas_handle,
                             rhs2_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                             rhs1_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                             n,
                             m,
                             k,
                             &alpha,
                             rhs2,
                             rhs2_stride,
                             rhs1,
                             rhs1_stride,
                             &beta,
                             lhs,
                             lhs_stride));

    cudaStreamDestroy(task_stream);
  }
};

template <>
struct MatMulImplBody<VariantKind::GPU, LegateTypeCode::HALF_LT> {
  void operator()(size_t m,
                  size_t n,
                  size_t k,
                  float* lhs,
                  const __half* rhs1,
                  const __half* rhs2,
                  size_t lhs_stride,
                  size_t rhs1_stride,
                  size_t rhs2_stride,
                  bool rhs1_transposed,
                  bool rhs2_transposed)
  {
    cublasHandle_t cublas_handle = Core::get_cublas();
    // Update the stream because the CUDA hijack can't see inside cuBLAS
    cudaStream_t task_stream;
    cudaStreamCreate(&task_stream);
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const float alpha = 1.f;
    const float beta  = 0.f;

    // cublas is dumb and doesn't support row-major, so reverse the matrix
    // order to help cublas think things are column-major
    // effectively we get NxM = NxK * KxM
    // Use the extended sgemm interface so we can use tensor cores
    // if they are available for this matrix shape and GPU
    CHECK_CUBLAS(cublasSgemmEx(cublas_handle,
                               rhs2_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                               rhs1_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                               n,
                               m,
                               k,
                               &alpha,
                               rhs2,
                               CUDA_R_16F,
                               rhs2_stride,
                               rhs1,
                               CUDA_R_16F,
                               rhs1_stride,
                               &beta,
                               lhs,
                               CUDA_R_32F,
                               lhs_stride));

    cudaStreamDestroy(task_stream);
  }
};

/*static*/ void MatMulTask::gpu_variant(TaskContext& context)
{
  matmul_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
