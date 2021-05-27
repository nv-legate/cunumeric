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

#include "matrix/matmul.h"
#include "matrix/matmul_template.inl"

#include "cuda_help.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <>
struct MatMulImplBody<VariantKind::GPU, LegateTypeCode::FLOAT_LT> {
  void operator()(size_t m,
                  size_t n,
                  size_t k,
                  float *lhs,
                  const float *rhs1,
                  const float *rhs2,
                  size_t lhs_stride,
                  size_t rhs1_stride,
                  size_t rhs2_stride)
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
                               CUBLAS_OP_N,
                               CUBLAS_OP_N,
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
                  double *lhs,
                  const double *rhs1,
                  const double *rhs2,
                  size_t lhs_stride,
                  size_t rhs1_stride,
                  size_t rhs2_stride)
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
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
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
  template <typename LHS>
  void operator()(size_t m,
                  size_t n,
                  size_t k,
                  LHS *lhs,
                  const __half *rhs1,
                  const __half *rhs2,
                  size_t lhs_stride,
                  size_t rhs1_stride,
                  size_t rhs2_stride)
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
                               CUBLAS_OP_N,
                               CUBLAS_OP_N,
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
                               sizeof(LHS) == sizeof(float) ? CUDA_R_32F : CUDA_R_16F,
                               lhs_stride));

    cudaStreamDestroy(task_stream);
  }
};

/*static*/ void MatMulTask::gpu_variant(const Task *task,
                                        const std::vector<PhysicalRegion> &regions,
                                        Context context,
                                        Runtime *runtime)
{
  matmul_template<VariantKind::GPU>(task, regions, context, runtime);
}

}  // namespace numpy
}  // namespace legate
