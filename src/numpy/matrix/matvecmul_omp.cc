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

#include "numpy/matrix/matvecmul.h"
#include "numpy/matrix/matvecmul_template.inl"
#include "numpy/matrix/util.h"
#include "numpy/matrix/util_omp.h"

#include <cblas.h>
#include <omp.h>

namespace legate {
namespace numpy {

using namespace Legion;

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

    auto mat_copy = allocate_buffer(m * n);
    auto vec_copy = allocate_buffer(vec_size);

    half_matrix_to_float_omp(mat_copy, mat, m, n, mat_stride);
    half_vector_to_float_omp(vec_copy, vec, vec_size);

    auto trans = transpose_mat ? CblasTrans : CblasNoTrans;
    cblas_sgemv(CblasRowMajor, trans, m, n, 1, mat_copy, n, vec_copy, 1, 0, lhs, 1);
  }
};

/*static*/ void MatVecMulTask::omp_variant(TaskContext& context)
{
  openblas_set_num_threads(omp_get_max_threads());
  matvecmul_template<VariantKind::OMP>(context);
}

}  // namespace numpy
}  // namespace legate
