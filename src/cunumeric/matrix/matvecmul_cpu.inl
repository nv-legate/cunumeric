/* Copyright 2023 NVIDIA Corporation
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

#pragma once

#include "cunumeric/matrix/matvecmul.h"
#include "cunumeric/matrix/matvecmul_template.inl"
#include "cunumeric/matrix/util.h"

#include <cblas.h>

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND>
struct MatVecMulImplBody<KIND, LegateTypeCode::FLOAT_LT> {
  void operator()(size_t m,
                  size_t n,
                  float* lhs,
                  const float* mat,
                  const float* vec,
                  size_t mat_stride,
                  bool transpose_mat,
                  bool lhs_readable)
  {
    auto trans = transpose_mat ? CblasTrans : CblasNoTrans;
    // lhs_readable being true means that the matvecmul tasks can overwrite the lhs
    float beta = lhs_readable ? 0.0 : 1.0;
    cblas_sgemv(CblasRowMajor, trans, m, n, 1, mat, mat_stride, vec, 1, beta, lhs, 1);
  }
};

template <VariantKind KIND>
struct MatVecMulImplBody<KIND, LegateTypeCode::DOUBLE_LT> {
  void operator()(size_t m,
                  size_t n,
                  double* lhs,
                  const double* mat,
                  const double* vec,
                  size_t mat_stride,
                  bool transpose_mat,
                  bool lhs_readable)
  {
    auto trans  = transpose_mat ? CblasTrans : CblasNoTrans;
    double beta = lhs_readable ? 0.0 : 1.0;
    cblas_dgemv(CblasRowMajor, trans, m, n, 1, mat, mat_stride, vec, 1, beta, lhs, 1);
  }
};

template <VariantKind KIND>
struct MatVecMulImplBody<KIND, LegateTypeCode::HALF_LT> {
  void operator()(size_t m,
                  size_t n,
                  float* lhs,
                  const __half* mat,
                  const __half* vec,
                  size_t mat_stride,
                  bool transpose_mat,
                  bool lhs_readable)
  {
    auto vec_size = transpose_mat ? m : n;

    auto mat_copy = allocate_buffer(m * n);
    auto vec_copy = allocate_buffer(vec_size);

    half_matrix_to_float(mat_copy, mat, m, n, mat_stride);
    half_vector_to_float(vec_copy, vec, vec_size);

    MatVecMulImplBody<KIND, LegateTypeCode::FLOAT_LT>{}(
      m, n, lhs, mat_copy, vec_copy, n, transpose_mat, lhs_readable);
  }
};

template <VariantKind KIND>
struct MatVecMulImplBody<KIND, LegateTypeCode::COMPLEX64_LT> {
  void operator()(size_t m,
                  size_t n,
                  complex<float>* lhs_,
                  const complex<float>* mat_,
                  const complex<float>* vec_,
                  size_t mat_stride,
                  bool transpose_mat,
                  bool lhs_readable)
  {
    __complex__ float* lhs       = reinterpret_cast<__complex__ float*>(lhs_);
    const __complex__ float* mat = reinterpret_cast<const __complex__ float*>(mat_);
    const __complex__ float* vec = reinterpret_cast<const __complex__ float*>(vec_);
    __complex__ float alpha      = 1.0;
    __complex__ float beta       = lhs_readable ? 0.0 : 1.0;

    auto trans = transpose_mat ? CblasTrans : CblasNoTrans;
    cblas_cgemv(CblasRowMajor, trans, m, n, &alpha, mat, mat_stride, vec, 1, &beta, lhs, 1);
  }
};

template <VariantKind KIND>
struct MatVecMulImplBody<KIND, LegateTypeCode::COMPLEX128_LT> {
  void operator()(size_t m,
                  size_t n,
                  complex<double>* lhs_,
                  const complex<double>* mat_,
                  const complex<double>* vec_,
                  size_t mat_stride,
                  bool transpose_mat,
                  bool lhs_readable)
  {
    __complex__ double* lhs       = reinterpret_cast<__complex__ double*>(lhs_);
    const __complex__ double* mat = reinterpret_cast<const __complex__ double*>(mat_);
    const __complex__ double* vec = reinterpret_cast<const __complex__ double*>(vec_);
    __complex__ double alpha      = 1.0;
    __complex__ double beta       = lhs_readable ? 0.0 : 1.0;

    auto trans = transpose_mat ? CblasTrans : CblasNoTrans;
    cblas_zgemv(CblasRowMajor, trans, m, n, &alpha, mat, mat_stride, vec, 1, &beta, lhs, 1);
  }
};

}  // namespace cunumeric
