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
#include "cunumeric/matrix/util.h"
#include "cunumeric/matrix/util_omp.h"

#include <cblas.h>
#include <omp.h>

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <>
struct MatMulImplBody<VariantKind::OMP, LegateTypeCode::FLOAT_LT> {
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
                0,
                lhs,
                lhs_stride);
  }
};

template <>
struct MatMulImplBody<VariantKind::OMP, LegateTypeCode::DOUBLE_LT> {
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
                0,
                lhs,
                lhs_stride);
  }
};

template <>
struct MatMulImplBody<VariantKind::OMP, LegateTypeCode::HALF_LT> {
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
    auto rhs1_copy = allocate_buffer(m * k);
    auto rhs2_copy = allocate_buffer(k * n);

    if (rhs1_transposed)
      half_matrix_to_float_omp(rhs1_copy, rhs1, k, m, rhs1_stride);
    else
      half_matrix_to_float_omp(rhs1_copy, rhs1, m, k, rhs1_stride);

    if (rhs2_transposed)
      half_matrix_to_float_omp(rhs2_copy, rhs2, n, k, rhs2_stride);
    else
      half_matrix_to_float_omp(rhs2_copy, rhs2, k, n, rhs2_stride);

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
                0,
                lhs,
                lhs_stride);
  }
};

/*static*/ void MatMulTask::omp_variant(TaskContext& context)
{
  openblas_set_num_threads(omp_get_max_threads());
  matmul_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
