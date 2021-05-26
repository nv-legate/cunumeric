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

#include <cblas.h>
#include <omp.h>

namespace legate {
namespace numpy {

using namespace Legion;

template <>
struct MatMulImplBody<VariantKind::OMP, LegateTypeCode::FLOAT_LT> {
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
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
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
                  double *lhs,
                  const double *rhs1,
                  const double *rhs2,
                  size_t lhs_stride,
                  size_t rhs1_stride,
                  size_t rhs2_stride)
  {
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
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

/*static*/ void MatMulTask::omp_variant(const Task *task,
                                        const std::vector<PhysicalRegion> &regions,
                                        Context context,
                                        Runtime *runtime)
{
  openblas_set_num_threads(omp_get_max_threads());
  matmul_template<VariantKind::OMP>(task, regions, context, runtime);
}

}  // namespace numpy
}  // namespace legate
