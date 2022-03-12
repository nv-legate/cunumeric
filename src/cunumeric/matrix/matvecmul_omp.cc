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
#include "cunumeric/matrix/util_omp.h"

#include <cblas.h>
#include <omp.h>

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <>
struct MatVecMulImplBody<VariantKind::OMP, LegateTypeCode::FLOAT_LT> {
  void operator()(size_t m,
                  size_t n,
                  float* lhs,
                  const float* mat,
                  const float* vec,
                  size_t mat_stride,
                  bool transpose_mat)
  {
    auto trans = transpose_mat ? CblasTrans : CblasNoTrans;
    cblas_sgemv(CblasRowMajor, trans, m, n, 1, mat, mat_stride, vec, 1, 0, lhs, 1);
  }
};

template <>
struct MatVecMulImplBody<VariantKind::OMP, LegateTypeCode::DOUBLE_LT> {
  void operator()(size_t m,
                  size_t n,
                  double* lhs,
                  const double* mat,
                  const double* vec,
                  size_t mat_stride,
                  bool transpose_mat)
  {
    auto trans = transpose_mat ? CblasTrans : CblasNoTrans;
    cblas_dgemv(CblasRowMajor, trans, m, n, 1, mat, mat_stride, vec, 1, 0, lhs, 1);
  }
};

template <>
struct MatVecMulImplBody<VariantKind::OMP, LegateTypeCode::HALF_LT> {
  void operator()(size_t m,
                  size_t n,
                  float* lhs,
                  const __half* mat,
                  const __half* vec,
                  size_t mat_stride,
                  bool transpose_mat)
  {
    auto vec_size = transpose_mat ? m : n;

    auto mat_copy = allocate_buffer_omp(m * n);
    auto vec_copy = allocate_buffer_omp(vec_size);

    half_matrix_to_float_omp(mat_copy, mat, m, n, mat_stride);
    half_vector_to_float_omp(vec_copy, vec, vec_size);

    MatVecMulImplBody<VariantKind::OMP, LegateTypeCode::FLOAT_LT>{}(
      m, n, lhs, mat_copy, vec_copy, n, transpose_mat);
  }
};

template <>
struct MatVecMulImplBody<VariantKind::OMP, LegateTypeCode::COMPLEX64_LT> {
  void operator()(size_t m,
                  size_t n,
                  complex<float>* lhs_,
                  const complex<float>* mat_,
                  const complex<float>* vec_,
                  size_t mat_stride,
                  bool transpose_mat)
  {
    __complex__ float* lhs       = reinterpret_cast<__complex__ float*>(lhs_);
    const __complex__ float* mat = reinterpret_cast<const __complex__ float*>(mat_);
    const __complex__ float* vec = reinterpret_cast<const __complex__ float*>(vec_);
    __complex__ float alpha      = 1.0;
    __complex__ float beta       = 0.0;

    auto trans = transpose_mat ? CblasTrans : CblasNoTrans;
    cblas_cgemv(CblasRowMajor, trans, m, n, &alpha, mat, mat_stride, vec, 1, &beta, lhs, 1);
  }
};

template <>
struct MatVecMulImplBody<VariantKind::OMP, LegateTypeCode::COMPLEX128_LT> {
  void operator()(size_t m,
                  size_t n,
                  complex<double>* lhs_,
                  const complex<double>* mat_,
                  const complex<double>* vec_,
                  size_t mat_stride,
                  bool transpose_mat)
  {
    __complex__ double* lhs       = reinterpret_cast<__complex__ double*>(lhs_);
    const __complex__ double* mat = reinterpret_cast<const __complex__ double*>(mat_);
    const __complex__ double* vec = reinterpret_cast<const __complex__ double*>(vec_);
    __complex__ double alpha      = 1.0;
    __complex__ double beta       = 0.0;

    auto trans = transpose_mat ? CblasTrans : CblasNoTrans;
    cblas_zgemv(CblasRowMajor, trans, m, n, &alpha, mat, mat_stride, vec, 1, &beta, lhs, 1);
  }
};

/*static*/ void MatVecMulTask::omp_variant(TaskContext& context)
{
  openblas_set_num_threads(omp_get_max_threads());
  matvecmul_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
