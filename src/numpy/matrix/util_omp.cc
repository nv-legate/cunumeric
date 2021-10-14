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

#include "legion.h"
#include "numpy/matrix/util_omp.h"

namespace legate {
namespace numpy {

using namespace Legion;

void half_vector_to_float_omp(float* out, const __half* ptr, size_t n)
{
#pragma omp parallel for schedule(static)
  for (size_t idx = 0; idx < n; idx++) out[idx] = ptr[idx];
}

void half_matrix_to_float_omp(float* out, const __half* ptr, size_t m, size_t n, size_t pitch)
{
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++) out[i * n + j] = ptr[i * pitch + j];
}

}  // namespace numpy
}  // namespace legate
