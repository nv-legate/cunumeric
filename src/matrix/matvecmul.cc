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
#include "matrix/util.h"

#include <cblas.h>
#ifdef LEGATE_USE_OPENMP
#include <omp.h>
#endif

namespace legate {
namespace numpy {

using namespace Legion;

template <>
struct MatVecMulImplBody<VariantKind::CPU, LegateTypeCode::FLOAT_LT> {
  void operator()(size_t m,
                  size_t n,
                  float *lhs,
                  const float *rhs1,
                  const float *rhs2,
                  size_t rhs_stride,
                  bool vec_on_lhs)
  {
    if (vec_on_lhs)
      cblas_sgemv(CblasRowMajor, CblasTrans, m, n, 1, rhs2, rhs_stride, rhs1, 1, 0, lhs, 1);
    else
      cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1, rhs1, rhs_stride, rhs2, 1, 0, lhs, 1);
  }
};

template <>
struct MatVecMulImplBody<VariantKind::CPU, LegateTypeCode::DOUBLE_LT> {
  void operator()(size_t m,
                  size_t n,
                  double *lhs,
                  const double *rhs1,
                  const double *rhs2,
                  size_t rhs_stride,
                  bool vec_on_lhs)
  {
    if (vec_on_lhs)
      cblas_dgemv(CblasRowMajor, CblasTrans, m, n, 1, rhs2, rhs_stride, rhs1, 1, 0, lhs, 1);
    else
      cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1, rhs1, rhs_stride, rhs2, 1, 0, lhs, 1);
  }
};

template <>
struct MatVecMulImplBody<VariantKind::CPU, LegateTypeCode::HALF_LT> {
  void operator()(size_t m,
                  size_t n,
                  __half *lhs,
                  const __half *rhs1,
                  const __half *rhs2,
                  size_t rhs_stride,
                  bool vec_on_lhs)
  {
    if (vec_on_lhs) {
      auto lhs_copy  = allocate_buffer(n);
      auto rhs1_copy = allocate_buffer(m);
      auto rhs2_copy = allocate_buffer(m * n);

      half_vector_to_float(rhs1_copy, rhs1, m);
      half_matrix_to_float(rhs2_copy, rhs2, m, n, rhs_stride);

      cblas_sgemv(
        CblasRowMajor, CblasTrans, m, n, 1, rhs2_copy, rhs_stride, rhs1_copy, 1, 0, lhs_copy, 1);

      float_vector_to_half(lhs, lhs_copy, n);
    } else {
      auto lhs_copy  = allocate_buffer(m);
      auto rhs1_copy = allocate_buffer(m * n);
      auto rhs2_copy = allocate_buffer(n);

      half_matrix_to_float(rhs1_copy, rhs1, m, n, rhs_stride);
      half_vector_to_float(rhs2_copy, rhs2, n);

      cblas_sgemv(
        CblasRowMajor, CblasNoTrans, m, n, 1, rhs1_copy, rhs_stride, rhs2_copy, 1, 0, lhs_copy, 1);

      float_vector_to_half(lhs, lhs_copy, m);
    }
  }

  void operator()(size_t m,
                  size_t n,
                  float *lhs,
                  const __half *rhs1,
                  const __half *rhs2,
                  size_t rhs_stride,
                  bool vec_on_lhs)
  {
    if (vec_on_lhs) {
      auto rhs1_copy = allocate_buffer(m);
      auto rhs2_copy = allocate_buffer(m * n);

      half_vector_to_float(rhs1_copy, rhs1, m);
      half_matrix_to_float(rhs2_copy, rhs2, m, n, rhs_stride);

      cblas_sgemv(
        CblasRowMajor, CblasTrans, m, n, 1, rhs2_copy, rhs_stride, rhs1_copy, 1, 0, lhs, 1);
    } else {
      auto rhs1_copy = allocate_buffer(m * n);
      auto rhs2_copy = allocate_buffer(n);

      half_matrix_to_float(rhs1_copy, rhs1, m, n, rhs_stride);
      half_vector_to_float(rhs2_copy, rhs2, n);

      cblas_sgemv(
        CblasRowMajor, CblasNoTrans, m, n, 1, rhs1_copy, rhs_stride, rhs2_copy, 1, 0, lhs, 1);
    }
  }
};

void deserialize(Deserializer &ctx, MatVecMulArgs &args)
{
  deserialize(ctx, args.needs_reduction);
  deserialize(ctx, args.lhs_shape);
  deserialize(ctx, args.rhs1_shape);
  deserialize(ctx, args.rhs2_shape);
  deserialize(ctx, args.lhs);
  deserialize(ctx, args.rhs1);
  deserialize(ctx, args.rhs2);
}

/*static*/ void MatVecMulTask::cpu_variant(const Task *task,
                                           const std::vector<PhysicalRegion> &regions,
                                           Context context,
                                           Runtime *runtime)
{
#ifdef LEGATE_USE_OPENMP
  openblas_set_num_threads(1);  // make sure this isn't overzealous
#endif
  matvecmul_template<VariantKind::CPU>(task, regions, context, runtime);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  MatVecMulTask::register_variants();
}
}  // namespace

}  // namespace numpy
}  // namespace legate
