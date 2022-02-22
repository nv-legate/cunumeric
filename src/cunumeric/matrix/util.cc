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
#include "cunumeric/matrix/util.h"

namespace cunumeric {

using namespace Legion;

int64_t calculate_volume(size_t ndim, const int64_t* shape, int64_t* strides)
{
  int64_t volume = 1;
  for (int d = ndim - 1; d >= 0; --d) {
    if (strides != nullptr) { strides[d] = volume; }
    volume *= shape[d];
  }
  return volume;
}

float* allocate_buffer(size_t size)
{
  Rect<1> bounds(0, size - 1);
  // We will not call this function on GPUs
  DeferredBuffer<float, 1> buffer(Memory::Kind::SYSTEM_MEM, bounds);
  return buffer.ptr(0);
}

void half_vector_to_float(float* out, const __half* ptr, size_t n)
{
  for (size_t idx = 0; idx < n; idx++) out[idx] = ptr[idx];
}

void float_vector_to_half(__half* out, const float* ptr, size_t n)
{
  for (size_t idx = 0; idx < n; idx++) out[idx] = ptr[idx];
}

void half_matrix_to_float(float* out, const __half* ptr, size_t m, size_t n, size_t pitch)
{
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++) out[i * n + j] = ptr[i * pitch + j];
}

void float_matrix_to_half(__half* out, const float* ptr, size_t m, size_t n, size_t pitch)
{
  for (unsigned i = 0; i < m; i++)
    for (unsigned j = 0; j < n; j++) out[i * pitch + j] = ptr[i * n + j];
}

void half_tensor_to_float(
  float* out, const __half* in, size_t ndim, const int64_t* shape, const int64_t* in_strides)
{
  int64_t volume = calculate_volume(ndim, shape);
  for (int64_t out_idx = 0; out_idx < volume; ++out_idx) {
    int64_t in_idx = unflatten_with_strides(out_idx, ndim, shape, in_strides);
    out[out_idx]   = in[in_idx];
  }
}

void float_tensor_to_half(
  __half* out, const float* in, size_t ndim, const int64_t* shape, const int64_t* out_strides)
{
  int64_t volume = calculate_volume(ndim, shape);
  for (int64_t in_idx = 0; in_idx < volume; ++in_idx) {
    int64_t out_idx = unflatten_with_strides(in_idx, ndim, shape, out_strides);
    out[out_idx]    = in[in_idx];
  }
}

}  // namespace cunumeric
