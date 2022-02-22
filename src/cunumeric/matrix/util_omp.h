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

#pragma once

#include "mathtypes/half.h"

namespace cunumeric {

float* allocate_buffer_omp(size_t size);

// The following assume that the float array was created using allocate_buffer

void half_vector_to_float_omp(float* out, const __half* ptr, size_t n);

void half_matrix_to_float_omp(float* out, const __half* ptr, size_t m, size_t n, size_t pitch);

void half_tensor_to_float_omp(
  float* out, const __half* in, size_t ndim, const int64_t* shape, const int64_t* in_strides);

void float_tensor_to_half_omp(
  __half* out, const float* in, size_t ndim, const int64_t* shape, const int64_t* out_strides);

}  // namespace cunumeric
