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

#include "legion.h"
#include "core/data/buffer.h"
#include "cunumeric/cunumeric.h"
#include "cunumeric/matrix/util.h"
#include "cunumeric/matrix/util_omp.h"

namespace cunumeric {

using namespace Legion;

float* allocate_buffer_omp(size_t size)
{
  Memory::Kind kind = CuNumeric::has_numamem ? Memory::Kind::SOCKET_MEM : Memory::Kind::SYSTEM_MEM;
  // We will not call this function on GPUs
  auto buffer = legate::create_buffer<float, 1>(size, kind);
  return buffer.ptr(0);
}

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

void half_tensor_to_float_omp(
  float* out, const __half* in, size_t ndim, const int64_t* shape, const int64_t* in_strides)
{
  int64_t volume = calculate_volume(ndim, shape);
#pragma omp parallel for schedule(static)
  for (int64_t out_idx = 0; out_idx < volume; ++out_idx) {
    int64_t in_idx = unflatten_with_strides(out_idx, ndim, shape, in_strides);
    out[out_idx]   = in[in_idx];
  }
}

void half_tensor_to_float_omp(
  float* out, const __half* in, size_t ndim, const tblis::len_type* shape, const tblis::stride_type* in_strides)
{
  // We're going to do some direct casts between TBLIS types and standard types.
  // To be sure that these are safe, assert that the sizes are the same.
  static_assert(sizeof(int64_t) == sizeof(tblis::len_type));
  static_assert(sizeof(int64_t) == sizeof(tblis::stride_type));
  return half_tensor_to_float_omp(out, in, ndim, reinterpret_cast<const int64_t*>(shape), reinterpret_cast<const int64_t*>(in_strides));
}

void float_tensor_to_half_omp(
  __half* out, const float* in, size_t ndim, const int64_t* shape, const int64_t* out_strides)
{
  int64_t volume = calculate_volume(ndim, shape);
#pragma omp parallel for schedule(static)
  for (int64_t in_idx = 0; in_idx < volume; ++in_idx) {
    int64_t out_idx = unflatten_with_strides(in_idx, ndim, shape, out_strides);
    out[out_idx]    = in[in_idx];
  }
}

void float_tensor_to_half_omp(
  __half* out, const float* in, size_t ndim, const tblis::len_type* shape, const tblis::stride_type* out_strides)
{
  // We're going to do some direct casts between TBLIS types and standard types.
  // To be sure that these are safe, assert that the sizes are the same.
  static_assert(sizeof(int64_t) == sizeof(tblis::len_type));
  static_assert(sizeof(int64_t) == sizeof(tblis::stride_type));
  return float_tensor_to_half_omp(out, in, ndim, reinterpret_cast<const int64_t*>(shape), reinterpret_cast<const int64_t*>(out_strides));
}

}  // namespace cunumeric
