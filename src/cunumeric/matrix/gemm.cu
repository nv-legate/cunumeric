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

#include "cunumeric/matrix/gemm.h"
#include "cunumeric/matrix/gemm_template.inl"

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace legate;

template <typename Gemm, typename VAL>
static inline void gemm_template(
  Gemm gemm, VAL* lhs, const VAL* rhs1, const VAL* rhs2, int32_t m, int32_t n, int32_t k)
{
  auto context = get_cublas();
  auto stream  = get_cached_stream();
  CHECK_CUBLAS(cublasSetStream(context, stream));

  auto transa = CUBLAS_OP_N;
  auto transb = CUBLAS_OP_T;

  VAL alpha = -1.0;
  VAL beta  = 1.0;

  CHECK_CUBLAS(gemm(context, transa, transb, m, n, k, &alpha, rhs1, m, rhs2, n, &beta, lhs, m));

  CHECK_CUDA_STREAM(stream);
}

template <typename Gemm, typename VAL, typename CTOR>
static inline void complex_gemm_template(
  Gemm gemm, VAL* lhs, const VAL* rhs1, const VAL* rhs2, int32_t m, int32_t n, int32_t k, CTOR ctor)
{
  auto context = get_cublas();
  auto stream  = get_cached_stream();
  CHECK_CUBLAS(cublasSetStream(context, stream));

  auto transa = CUBLAS_OP_N;
  auto transb = CUBLAS_OP_C;

  auto alpha = ctor(-1.0, 0.0);
  auto beta  = ctor(1.0, 0.0);

  CHECK_CUBLAS(gemm(context, transa, transb, m, n, k, &alpha, rhs1, m, rhs2, n, &beta, lhs, m));

  CHECK_CUDA_STREAM(stream);
}

template <>
struct GemmImplBody<VariantKind::GPU, LegateTypeCode::FLOAT_LT> {
  void operator()(float* lhs, const float* rhs1, const float* rhs2, int32_t m, int32_t n, int32_t k)
  {
    gemm_template(cublasSgemm, lhs, rhs1, rhs2, m, n, k);
  }
};

template <>
struct GemmImplBody<VariantKind::GPU, LegateTypeCode::DOUBLE_LT> {
  void operator()(
    double* lhs, const double* rhs1, const double* rhs2, int32_t m, int32_t n, int32_t k)
  {
    gemm_template(cublasDgemm, lhs, rhs1, rhs2, m, n, k);
  }
};

template <>
struct GemmImplBody<VariantKind::GPU, LegateTypeCode::COMPLEX64_LT> {
  void operator()(complex<float>* lhs_,
                  const complex<float>* rhs1_,
                  const complex<float>* rhs2_,
                  int32_t m,
                  int32_t n,
                  int32_t k)
  {
    auto lhs  = reinterpret_cast<cuComplex*>(lhs_);
    auto rhs1 = reinterpret_cast<const cuComplex*>(rhs1_);
    auto rhs2 = reinterpret_cast<const cuComplex*>(rhs2_);

    complex_gemm_template(cublasCgemm, lhs, rhs1, rhs2, m, n, k, make_float2);
  }
};

template <>
struct GemmImplBody<VariantKind::GPU, LegateTypeCode::COMPLEX128_LT> {
  void operator()(complex<double>* lhs_,
                  const complex<double>* rhs1_,
                  const complex<double>* rhs2_,
                  int32_t m,
                  int32_t n,
                  int32_t k)
  {
    auto lhs  = reinterpret_cast<cuDoubleComplex*>(lhs_);
    auto rhs1 = reinterpret_cast<const cuDoubleComplex*>(rhs1_);
    auto rhs2 = reinterpret_cast<const cuDoubleComplex*>(rhs2_);

    complex_gemm_template(cublasZgemm, lhs, rhs1, rhs2, m, n, k, make_double2);
  }
};

/*static*/ void GemmTask::gpu_variant(TaskContext& context)
{
  gemm_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
