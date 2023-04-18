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

#include "cunumeric/matrix/potrf.h"
#include "cunumeric/matrix/potrf_template.inl"

#include <cblas.h>
#include <lapack.h>
#include <omp.h>

namespace cunumeric {

using namespace legate;

template <>
struct PotrfImplBody<VariantKind::OMP, Type::FLOAT32> {
  void operator()(float* array, int32_t m, int32_t n)
  {
    char uplo    = 'L';
    int32_t info = 0;
    LAPACK_spotrf(&uplo, &n, array, &m, &info);
    if (info != 0) throw legate::TaskException("Matrix is not positive definite");
  }
};

template <>
struct PotrfImplBody<VariantKind::OMP, Type::FLOAT64> {
  void operator()(double* array, int32_t m, int32_t n)
  {
    char uplo    = 'L';
    int32_t info = 0;
    LAPACK_dpotrf(&uplo, &n, array, &m, &info);
    if (info != 0) throw legate::TaskException("Matrix is not positive definite");
  }
};

template <>
struct PotrfImplBody<VariantKind::OMP, Type::COMPLEX64> {
  void operator()(complex<float>* array, int32_t m, int32_t n)
  {
    char uplo    = 'L';
    int32_t info = 0;
    LAPACK_cpotrf(&uplo, &n, reinterpret_cast<__complex__ float*>(array), &m, &info);
    if (info != 0) throw legate::TaskException("Matrix is not positive definite");
  }
};

template <>
struct PotrfImplBody<VariantKind::OMP, Type::COMPLEX128> {
  void operator()(complex<double>* array, int32_t m, int32_t n)
  {
    char uplo    = 'L';
    int32_t info = 0;
    LAPACK_zpotrf(&uplo, &n, reinterpret_cast<__complex__ double*>(array), &m, &info);
    if (info != 0) throw legate::TaskException("Matrix is not positive definite");
  }
};

/*static*/ void PotrfTask::omp_variant(TaskContext& context)
{
  openblas_set_num_threads(omp_get_max_threads());
  potrf_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
