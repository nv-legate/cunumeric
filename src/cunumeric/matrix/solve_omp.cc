/* Copyright 2022 NVIDIA Corporation
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

#include "cunumeric/matrix/solve.h"
#include "cunumeric/matrix/solve_template.inl"

#include <cblas.h>
#include <lapack.h>

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <typename Getrf, typename Getrs, typename VAL>
static inline void solve_template(
  Getrf getrf, Getrs getrs, int32_t m, int32_t n, int32_t nrhs, VAL* a, VAL* b)
{
  const char trans = 'N';
  int32_t info     = 0;

  auto kind = CuNumeric::has_numamem ? Memory::Kind::SOCKET_MEM : Memory::Kind::SYSTEM_MEM;
  auto ipiv = create_buffer<int32_t>(std::min(m, n), kind);

  getrf(&m, &n, a, &m, ipiv.ptr(0), &info);
  getrs(&trans, &n, &nrhs, a, &m, ipiv.ptr(0), b, &n, &info);

  if (info != 0) throw legate::TaskException("Matrix is not positive definite");
}

template <>
struct SolveImplBody<VariantKind::OMP, LegateTypeCode::FLOAT_LT> {
  void operator()(int32_t m, int32_t n, int32_t nrhs, float* a, float* b)
  {
    solve_template(sgetrf_, sgetrs_, m, n, nrhs, a, b);
  }
};

template <>
struct SolveImplBody<VariantKind::OMP, LegateTypeCode::DOUBLE_LT> {
  void operator()(int32_t m, int32_t n, int32_t nrhs, double* a, double* b)
  {
    solve_template(dgetrf_, dgetrs_, m, n, nrhs, a, b);
  }
};

template <>
struct SolveImplBody<VariantKind::OMP, LegateTypeCode::COMPLEX64_LT> {
  void operator()(int32_t m, int32_t n, int32_t nrhs, complex<float>* a, complex<float>* b)
  {
    solve_template(cgetrf_,
                   cgetrs_,
                   m,
                   n,
                   nrhs,
                   reinterpret_cast<__complex__ float*>(a),
                   reinterpret_cast<__complex__ float*>(b));
  }
};

template <>
struct SolveImplBody<VariantKind::OMP, LegateTypeCode::COMPLEX128_LT> {
  void operator()(int32_t m, int32_t n, int32_t nrhs, complex<double>* a, complex<double>* b)
  {
    solve_template(zgetrf_,
                   zgetrs_,
                   m,
                   n,
                   nrhs,
                   reinterpret_cast<__complex__ double*>(a),
                   reinterpret_cast<__complex__ double*>(b));
  }
};

/*static*/ void SolveTask::omp_variant(TaskContext& context)
{
  solve_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
