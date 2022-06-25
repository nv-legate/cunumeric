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
#include "cunumeric/matrix/util.h"

namespace cunumeric {

using namespace Legion;

size_t stride_for_blas(size_t m, size_t n, size_t x_stride, size_t y_stride, bool& transpose)
{
  size_t blas_stride;
  if (n == 1) {
    // Column matrix: Every row has exactly 1 element, therefore it is trivially contiguous. Any
    // stride between rows is acceptable.
#ifdef DEBUG_CUNUMERIC
    assert(x_stride >= 1);
#endif
    blas_stride = x_stride;
    transpose   = false;
  } else if (m == 1) {
    // Row matrix
#ifdef DEBUG_CUNUMERIC
    assert(y_stride >= 1);
#endif
    if (y_stride == 1) {
      // Row elements are contiguous, so there is nothing to do. There is only one row, so the row
      // stride is irrelevant, but we have to pass a value >= n to appease the BLAS library.
      blas_stride = n;
      transpose   = false;
    } else {
      // If the elements in the row are s>1 slots apart, present the input to the BLAS library as a
      // row-major nx1 matrix with row stride equal to s, and ask for the matrix to be transposed.
      blas_stride = y_stride;
      transpose   = true;
    }
  } else {
    // General case: One dimension needs to be contiguous. If that's not the last dimension, then
    // the matrix represents the transpose of a row-major nxm matrix. We then tell the BLAS library
    // that we are passing a row-major nxm matrix, and ask for the matrix to be transposed.
#ifdef DEBUG_CUNUMERIC
    assert(x_stride == 1 && y_stride > 1 || y_stride == 1 && x_stride > 1);
#endif
    blas_stride = std::max(x_stride, y_stride);
    transpose   = x_stride == 1;
  }
  return blas_stride;
}

int64_t calculate_volume(size_t ndim, const int64_t* shape, int64_t* strides)
{
  int64_t volume = 1;
  for (int d = ndim - 1; d >= 0; --d) {
    if (strides != nullptr) { strides[d] = volume; }
    volume *= shape[d];
  }
  return volume;
}

int64_t calculate_volume(size_t ndim, const tblis::len_type* shape, tblis::stride_type* strides)
{
  // We're going to do some direct casts between TBLIS types and standard types.
  // To be sure that these are safe, assert that the sizes are the same.
  static_assert(sizeof(int64_t) == sizeof(tblis::len_type));
  static_assert(sizeof(int64_t) == sizeof(tblis::stride_type));
  return calculate_volume(ndim, reinterpret_cast<const int64_t*>(shape), reinterpret_cast<int64_t*>(strides));
}

float* allocate_buffer(size_t size)
{
  // We will not call this function on GPUs
  auto buffer = legate::create_buffer<float, 1>(size, Memory::Kind::SYSTEM_MEM);
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

void half_tensor_to_float(
  float* out, const __half* in, size_t ndim, const tblis::len_type* shape, const tblis::stride_type* in_strides)
{
  // We're going to do some direct casts between TBLIS types and standard types.
  // To be sure that these are safe, assert that the sizes are the same.
  static_assert(sizeof(int64_t) == sizeof(tblis::len_type));
  static_assert(sizeof(int64_t) == sizeof(tblis::stride_type));
  return half_tensor_to_float(out, in, ndim, reinterpret_cast<const int64_t*>(shape), reinterpret_cast<const int64_t*>(in_strides));
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

void float_tensor_to_half(
  __half* out, const float* in, size_t ndim, const tblis::len_type* shape, const tblis::stride_type* out_strides)
{
  // We're going to do some direct casts between TBLIS types and standard types.
  // To be sure that these are safe, assert that the sizes are the same.
  static_assert(sizeof(int64_t) == sizeof(tblis::len_type));
  static_assert(sizeof(int64_t) == sizeof(tblis::stride_type));
  return float_tensor_to_half(out, in, ndim, reinterpret_cast<const int64_t*>(shape), reinterpret_cast<const int64_t*>(out_strides));
}

}  // namespace cunumeric
