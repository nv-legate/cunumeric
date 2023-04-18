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

#include "cunumeric/matrix/matmul.h"
#include "cunumeric/matrix/matmul_template.inl"
#include "cunumeric/matrix/util.h"

#include <cblas.h>

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND>
struct MatMulImplBody<KIND, Type::FLOAT32> {
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
                  bool rhs2_transposed,
                  bool lhs_overwritable)
  {
    cblas_sgemm(CblasRowMajor,
                rhs1_transposed ? CblasTrans : CblasNoTrans,
                rhs2_transposed ? CblasTrans : CblasNoTrans,
                m,
                n,
                k,
                1,
                rhs1,
                rhs1_stride,
                rhs2,
                rhs2_stride,
                // lhs_overwritable being true means that the matmul tasks can overwrite the lhs
                lhs_overwritable ? 0 : 1,
                lhs,
                lhs_stride);
  }
};

template <VariantKind KIND>
struct MatMulImplBody<KIND, Type::FLOAT64> {
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
                  bool rhs2_transposed,
                  bool lhs_overwritable)
  {
    cblas_dgemm(CblasRowMajor,
                rhs1_transposed ? CblasTrans : CblasNoTrans,
                rhs2_transposed ? CblasTrans : CblasNoTrans,
                m,
                n,
                k,
                1,
                rhs1,
                rhs1_stride,
                rhs2,
                rhs2_stride,
                lhs_overwritable ? 0 : 1,
                lhs,
                lhs_stride);
  }
};

template <VariantKind KIND>
struct MatMulImplBody<KIND, Type::FLOAT16> {
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
                  bool rhs2_transposed,
                  bool lhs_overwritable)
  {
    auto rhs1_copy = allocate_buffer(m * k);
    auto rhs2_copy = allocate_buffer(k * n);

    if (rhs1_transposed)
      half_matrix_to_float(rhs1_copy, rhs1, k, m, rhs1_stride);
    else
      half_matrix_to_float(rhs1_copy, rhs1, m, k, rhs1_stride);

    if (rhs2_transposed)
      half_matrix_to_float(rhs2_copy, rhs2, n, k, rhs2_stride);
    else
      half_matrix_to_float(rhs2_copy, rhs2, k, n, rhs2_stride);

    cblas_sgemm(CblasRowMajor,
                rhs1_transposed ? CblasTrans : CblasNoTrans,
                rhs2_transposed ? CblasTrans : CblasNoTrans,
                m,
                n,
                k,
                1,
                rhs1_copy,
                rhs1_transposed ? m : k,
                rhs2_copy,
                rhs2_transposed ? k : n,
                lhs_overwritable ? 0 : 1,
                lhs,
                lhs_stride);
  }
};

template <VariantKind KIND>
struct MatMulImplBody<KIND, Type::COMPLEX64> {
  void operator()(size_t m,
                  size_t n,
                  size_t k,
                  complex<float>* lhs_,
                  const complex<float>* rhs1_,
                  const complex<float>* rhs2_,
                  size_t lhs_stride,
                  size_t rhs1_stride,
                  size_t rhs2_stride,
                  bool rhs1_transposed,
                  bool rhs2_transposed,
                  bool lhs_overwritable)
  {
    __complex__ float* lhs        = reinterpret_cast<__complex__ float*>(lhs_);
    const __complex__ float* rhs1 = reinterpret_cast<const __complex__ float*>(rhs1_);
    const __complex__ float* rhs2 = reinterpret_cast<const __complex__ float*>(rhs2_);
    __complex__ float alpha       = 1.0;
    __complex__ float beta        = lhs_overwritable ? 0.0 : 1.0;

    cblas_cgemm(CblasRowMajor,
                rhs1_transposed ? CblasTrans : CblasNoTrans,
                rhs2_transposed ? CblasTrans : CblasNoTrans,
                m,
                n,
                k,
                &alpha,
                rhs1,
                rhs1_stride,
                rhs2,
                rhs2_stride,
                &beta,
                lhs,
                lhs_stride);
  }
};

template <VariantKind KIND>
struct MatMulImplBody<KIND, Type::COMPLEX128> {
  void operator()(size_t m,
                  size_t n,
                  size_t k,
                  complex<double>* lhs_,
                  const complex<double>* rhs1_,
                  const complex<double>* rhs2_,
                  size_t lhs_stride,
                  size_t rhs1_stride,
                  size_t rhs2_stride,
                  bool rhs1_transposed,
                  bool rhs2_transposed,
                  bool lhs_overwritable)
  {
    __complex__ double* lhs        = reinterpret_cast<__complex__ double*>(lhs_);
    const __complex__ double* rhs1 = reinterpret_cast<const __complex__ double*>(rhs1_);
    const __complex__ double* rhs2 = reinterpret_cast<const __complex__ double*>(rhs2_);
    __complex__ double alpha       = 1.0;
    __complex__ double beta        = lhs_overwritable ? 0.0 : 1.0;

    cblas_zgemm(CblasRowMajor,
                rhs1_transposed ? CblasTrans : CblasNoTrans,
                rhs2_transposed ? CblasTrans : CblasNoTrans,
                m,
                n,
                k,
                &alpha,
                rhs1,
                rhs1_stride,
                rhs2,
                rhs2_stride,
                &beta,
                lhs,
                lhs_stride);
  }
};

}  // namespace cunumeric
