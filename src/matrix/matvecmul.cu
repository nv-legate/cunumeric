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

#include "matrix/matvecmul.h"
#include "matrix/matvecmul_template.inl"

#include "cuda_help.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <>
struct MatVecMulImplBody<VariantKind::GPU, LegateTypeCode::FLOAT_LT> {
  void operator()(size_t m,
                  size_t n,
                  float *lhs,
                  const float *rhs1,
                  const float *rhs2,
                  size_t rhs_stride,
                  bool vec_on_lhs)
  {
    cublasHandle_t cublas_handle = Core::get_cublas();
    // Update the stream because the CUDA hijack can't see inside cuBLAS
    cudaStream_t task_stream;
    cudaStreamCreate(&task_stream);
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const float alpha = 1.f;
    const float beta  = 0.f;

    if (vec_on_lhs) {
      CHECK_CUBLAS(cublasSgemv(
        cublas_handle, CUBLAS_OP_N, n, m, &alpha, rhs2, rhs_stride, rhs1, 1, &beta, lhs, 1));
    } else {
      CHECK_CUBLAS(cublasSgemv(
        cublas_handle, CUBLAS_OP_T, n, m, &alpha, rhs1, rhs_stride, rhs2, 1, &beta, lhs, 1));
    }

    cudaStreamDestroy(task_stream);
  }
};

template <>
struct MatVecMulImplBody<VariantKind::GPU, LegateTypeCode::DOUBLE_LT> {
  void operator()(size_t m,
                  size_t n,
                  double *lhs,
                  const double *rhs1,
                  const double *rhs2,
                  size_t rhs_stride,
                  bool vec_on_lhs)
  {
    cublasHandle_t cublas_handle = Core::get_cublas();
    // Update the stream because the CUDA hijack can't see inside cuBLAS
    cudaStream_t task_stream;
    cudaStreamCreate(&task_stream);
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const double alpha = 1.f;
    const double beta  = 0.f;

    if (vec_on_lhs) {
      CHECK_CUBLAS(cublasDgemv(
        cublas_handle, CUBLAS_OP_N, n, m, &alpha, rhs2, rhs_stride, rhs1, 1, &beta, lhs, 1));
    } else {
      CHECK_CUBLAS(cublasDgemv(
        cublas_handle, CUBLAS_OP_T, n, m, &alpha, rhs1, rhs_stride, rhs2, 1, &beta, lhs, 1));
    }

    cudaStreamDestroy(task_stream);
  }
};

/*static*/ void MatVecMulTask::gpu_variant(const Task *task,
                                           const std::vector<PhysicalRegion> &regions,
                                           Context context,
                                           Runtime *runtime)
{
  matvecmul_template<VariantKind::GPU>(task, regions, context, runtime);
}

}  // namespace numpy
}  // namespace legate
