/* Copyright 2021-2022 NVIDIA Corporation
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
    auto cublas_handle = get_cublas();
    auto task_stream   = get_cached_stream();
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const float alpha = 1.0;
    const float beta  = 0.0;

    auto trans = transpose_mat ? CUBLAS_OP_N : CUBLAS_OP_T;
#if CUDART_VERSION >= 11040
    CHECK_CUBLAS(
      cublasSgemv(cublas_handle, trans, n, m, &alpha, mat, mat_stride, vec, 1, &beta, lhs, 1));
#else
    CHECK_CUBLAS(cublasSgemmEx(cublas_handle,
                               trans,
                               CUBLAS_OP_N,
                               transpose_mat ? n : m,
                               1,
                               transpose_mat ? m : n,
                               &alpha,
                               mat,
                               CUDA_R_32F,
                               mat_stride,
                               vec,
                               CUDA_R_32F,
                               transpose_mat ? m : n,
                               &beta,
                               lhs,
                               CUDA_R_32F,
                               transpose_mat ? n : m));
#endif
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
    auto cublas_handle = get_cublas();
    auto task_stream   = get_cached_stream();
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const double alpha = 1.0;
    const double beta  = 0.0;

    auto trans = transpose_mat ? CUBLAS_OP_N : CUBLAS_OP_T;
#if CUDART_VERSION >= 11040
    CHECK_CUBLAS(
      cublasDgemv(cublas_handle, trans, n, m, &alpha, mat, mat_stride, vec, 1, &beta, lhs, 1));
#else
    CHECK_CUBLAS(cublasDgemm(cublas_handle,
                             trans,
                             CUBLAS_OP_N,
                             transpose_mat ? n : m,
                             1,
                             transpose_mat ? m : n,
                             &alpha,
                             mat,
                             mat_stride,
                             vec,
                             transpose_mat ? m : n,
                             &beta,
                             lhs,
                             transpose_mat ? n : m));
#endif
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
    auto cublas_handle = get_cublas();
    auto task_stream   = get_cached_stream();
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const float alpha = 1.0;
    const float beta  = 0.0;

    auto trans = transpose_mat ? CUBLAS_OP_N : CUBLAS_OP_T;
    // Use SgemmEx here since there is no half precision gemv yet
    CHECK_CUBLAS(cublasSgemmEx(cublas_handle,
                               trans,
                               CUBLAS_OP_N,
                               transpose_mat ? n : m,
                               1,
                               transpose_mat ? m : n,
                               &alpha,
                               mat,
                               CUDA_R_16F,
                               mat_stride,
                               vec,
                               CUDA_R_16F,
                               transpose_mat ? m : n,
                               &beta,
                               lhs,
                               CUDA_R_32F,
                               transpose_mat ? n : m));
  }
};

template <>
struct MatVecMulImplBody<VariantKind::GPU, LegateTypeCode::COMPLEX64_LT> {
  void operator()(size_t m,
                  size_t n,
                  complex<float>* lhs_,
                  const complex<float>* mat_,
                  const complex<float>* vec_,
                  size_t mat_stride,
                  bool transpose_mat)
  {
    cuComplex* lhs       = reinterpret_cast<cuComplex*>(lhs_);
    const cuComplex* mat = reinterpret_cast<const cuComplex*>(mat_);
    const cuComplex* vec = reinterpret_cast<const cuComplex*>(vec_);

    auto cublas_handle = get_cublas();
    auto task_stream   = get_cached_stream();
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const cuComplex alpha = make_float2(1.0, 0.0);
    const cuComplex beta  = make_float2(0.0, 0.0);

    auto trans = transpose_mat ? CUBLAS_OP_N : CUBLAS_OP_T;
#if CUDART_VERSION >= 11040
    CHECK_CUBLAS(
      cublasCgemv(cublas_handle, trans, n, m, &alpha, mat, mat_stride, vec, 1, &beta, lhs, 1));
#else
    CHECK_CUBLAS(cublasCgemmEx(cublas_handle,
                               trans,
                               CUBLAS_OP_N,
                               transpose_mat ? n : m,
                               1,
                               transpose_mat ? m : n,
                               &alpha,
                               mat,
                               CUDA_C_32F,
                               mat_stride,
                               vec,
                               CUDA_C_32F,
                               transpose_mat ? m : n,
                               &beta,
                               lhs,
                               CUDA_C_32F,
                               transpose_mat ? n : m));
#endif
  }
};

template <>
struct MatVecMulImplBody<VariantKind::GPU, LegateTypeCode::COMPLEX128_LT> {
  void operator()(size_t m,
                  size_t n,
                  complex<double>* lhs_,
                  const complex<double>* mat_,
                  const complex<double>* vec_,
                  size_t mat_stride,
                  bool transpose_mat)
  {
    cuDoubleComplex* lhs       = reinterpret_cast<cuDoubleComplex*>(lhs_);
    const cuDoubleComplex* mat = reinterpret_cast<const cuDoubleComplex*>(mat_);
    const cuDoubleComplex* vec = reinterpret_cast<const cuDoubleComplex*>(vec_);

    auto cublas_handle = get_cublas();
    auto task_stream   = get_cached_stream();
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const cuDoubleComplex alpha = make_double2(1.0, 0.0);
    const cuDoubleComplex beta  = make_double2(0.0, 0.0);

    auto trans = transpose_mat ? CUBLAS_OP_N : CUBLAS_OP_T;
#if CUDART_VERSION >= 11040
    CHECK_CUBLAS(
      cublasZgemv(cublas_handle, trans, n, m, &alpha, mat, mat_stride, vec, 1, &beta, lhs, 1));
#else
    CHECK_CUBLAS(cublasZgemm(cublas_handle,
                             trans,
                             CUBLAS_OP_N,
                             transpose_mat ? n : m,
                             1,
                             transpose_mat ? m : n,
                             &alpha,
                             mat,
                             mat_stride,
                             vec,
                             transpose_mat ? m : n,
                             &beta,
                             lhs,
                             transpose_mat ? n : m));
#endif
  }
};

/*static*/ void MatVecMulTask::gpu_variant(TaskContext& context)
{
  matvecmul_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
