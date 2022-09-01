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

template <>
struct SolveImplBody<VariantKind::CPU, LegateTypeCode::FLOAT_LT> {
  void operator()(int32_t m, int32_t n, int32_t nrhs, float* a, float* b)
  {
    const char trans = 'N';
    int32_t info     = 0;

    auto kind = CuNumeric::has_numamem ? Memory::Kind::SOCKET_MEM : Memory::Kind::SYSTEM_MEM;
    auto ipiv = create_buffer<int32_t>(std::min(m, n), kind);

    LAPACK_sgetrf(&m, &n, a, &m, ipiv.ptr(0), &info);
    LAPACK_sgetrs(&trans, &n, &nrhs, a, &m, ipiv.ptr(0), b, &n, &info);

    if (info != 0) throw legate::TaskException("Matrix is not positive definite");
  }
};

template <>
struct SolveImplBody<VariantKind::CPU, LegateTypeCode::DOUBLE_LT> {
  void operator()(int32_t m, int32_t n, int32_t nrhs, double* a, double* b)
  {
    const char trans = 'N';
    int32_t info     = 0;

    auto kind = CuNumeric::has_numamem ? Memory::Kind::SOCKET_MEM : Memory::Kind::SYSTEM_MEM;
    auto ipiv = create_buffer<int32_t>(std::min(m, n), kind);

    LAPACK_dgetrf(&m, &n, a, &m, ipiv.ptr(0), &info);
    LAPACK_dgetrs(&trans, &n, &nrhs, a, &m, ipiv.ptr(0), b, &n, &info);

    if (info != 0) throw legate::TaskException("Matrix is not positive definite");
  }
};

template <>
struct SolveImplBody<VariantKind::CPU, LegateTypeCode::COMPLEX64_LT> {
  void operator()(int32_t m, int32_t n, int32_t nrhs, complex<float>* a_, complex<float>* b_)
  {
    const char trans = 'N';
    int32_t info     = 0;

    auto kind = CuNumeric::has_numamem ? Memory::Kind::SOCKET_MEM : Memory::Kind::SYSTEM_MEM;
    auto ipiv = create_buffer<int32_t>(std::min(m, n), kind);

    auto a = reinterpret_cast<__complex__ float*>(a_);
    auto b = reinterpret_cast<__complex__ float*>(b_);

    LAPACK_cgetrf(&m, &n, a, &m, ipiv.ptr(0), &info);
    LAPACK_cgetrs(&trans, &n, &nrhs, a, &m, ipiv.ptr(0), b, &n, &info);

    if (info != 0) throw legate::TaskException("Matrix is not positive definite");
  }
};

template <>
struct SolveImplBody<VariantKind::CPU, LegateTypeCode::COMPLEX128_LT> {
  void operator()(int32_t m, int32_t n, int32_t nrhs, complex<double>* a_, complex<double>* b_)
  {
    const char trans = 'N';
    int32_t info     = 0;

    auto kind = CuNumeric::has_numamem ? Memory::Kind::SOCKET_MEM : Memory::Kind::SYSTEM_MEM;
    auto ipiv = create_buffer<int32_t>(std::min(m, n), kind);

    auto a = reinterpret_cast<__complex__ double*>(a_);
    auto b = reinterpret_cast<__complex__ double*>(b_);

    LAPACK_zgetrf(&m, &n, a, &m, ipiv.ptr(0), &info);
    LAPACK_zgetrs(&trans, &n, &nrhs, a, &m, ipiv.ptr(0), b, &n, &info);

    if (info != 0) throw legate::TaskException("Matrix is not positive definite");
  }
};

/*static*/ void SolveTask::cpu_variant(TaskContext& context)
{
#ifdef LEGATE_USE_OPENMP
  openblas_set_num_threads(1);  // make sure this isn't overzealous
#endif
  solve_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { SolveTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
