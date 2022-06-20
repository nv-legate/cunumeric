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

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <typename Potrf, typename VAL>
static inline void potrf_template(Potrf potrf, VAL* array, int32_t m, int32_t n)
{
  char uplo    = 'L';
  int32_t info = 0;
  potrf(&uplo, &n, array, &m, &info);
  if (info != 0) throw legate::TaskException("Matrix is not positive definite");
}

template <>
struct PotrfImplBody<VariantKind::CPU, LegateTypeCode::FLOAT_LT> {
  void operator()(float* array, int32_t m, int32_t n) { potrf_template(spotrf_, array, m, n); }
};

template <>
struct PotrfImplBody<VariantKind::CPU, LegateTypeCode::DOUBLE_LT> {
  void operator()(double* array, int32_t m, int32_t n) { potrf_template(dpotrf_, array, m, n); }
};

template <>
struct PotrfImplBody<VariantKind::CPU, LegateTypeCode::COMPLEX64_LT> {
  void operator()(complex<float>* array, int32_t m, int32_t n)
  {
    potrf_template(cpotrf_, reinterpret_cast<__complex__ float*>(array), m, n);
  }
};

template <>
struct PotrfImplBody<VariantKind::CPU, LegateTypeCode::COMPLEX128_LT> {
  void operator()(complex<double>* array, int32_t m, int32_t n)
  {
    potrf_template(zpotrf_, reinterpret_cast<__complex__ double*>(array), m, n);
  }
};

/*static*/ void PotrfTask::cpu_variant(TaskContext& context)
{
#ifdef LEGATE_USE_OPENMP
  openblas_set_num_threads(1);  // make sure this isn't overzealous
#endif
  potrf_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { PotrfTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
