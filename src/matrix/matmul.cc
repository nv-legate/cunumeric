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
#include "matrix/util.h"

#include <cblas.h>
#ifdef LEGATE_USE_OPENMP
#include <omp.h>
#endif

namespace legate {
namespace numpy {

using namespace Legion;

template <>
struct MatMulImplBody<VariantKind::CPU, LegateTypeCode::FLOAT_LT> {
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
struct MatMulImplBody<VariantKind::CPU, LegateTypeCode::DOUBLE_LT> {
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

template <>
struct MatMulImplBody<VariantKind::CPU, LegateTypeCode::HALF_LT> {
  void operator()(size_t m,
                  size_t n,
                  size_t k,
                  __half *lhs,
                  const __half *rhs1,
                  const __half *rhs2,
                  size_t lhs_stride,
                  size_t rhs1_stride,
                  size_t rhs2_stride)
  {
    auto rhs1_copy = allocate_buffer(m * k);
    auto rhs2_copy = allocate_buffer(k * n);
    auto lhs_copy  = allocate_buffer(m * n);

    half_matrix_to_float(rhs1_copy, rhs1, m, k, rhs1_stride);
    half_matrix_to_float(rhs2_copy, rhs2, k, n, rhs2_stride);

    MatMulImplBody<VariantKind::CPU, LegateTypeCode::FLOAT_LT>()(
      m, n, k, lhs_copy, rhs1_copy, rhs2_copy, n, k, n);

    float_matrix_to_half(lhs, lhs_copy, m, n, lhs_stride);
  }
};

void deserialize(Deserializer &ctx, MatMulArgs &args)
{
  deserialize(ctx, args.needs_reduction);
  deserialize(ctx, args.lhs_shape);
  deserialize(ctx, args.rhs1_shape);
  deserialize(ctx, args.rhs2_shape);
  deserialize(ctx, args.lhs);
  deserialize(ctx, args.rhs1);
  deserialize(ctx, args.rhs2);
}

/*static*/ void MatMulTask::cpu_variant(const Task *task,
                                        const std::vector<PhysicalRegion> &regions,
                                        Context context,
                                        Runtime *runtime)
{
#ifdef LEGATE_USE_OPENMP
  openblas_set_num_threads(1);  // make sure this isn't overzealous
#endif
  matmul_template<VariantKind::CPU>(task, regions, context, runtime);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { MatMulTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
