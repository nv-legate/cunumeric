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

#pragma once

#include "mathtypes/half.h"

namespace cunumeric {

float* allocate_buffer(size_t size);

void half_vector_to_float(float* out, const __half* ptr, size_t n);

void float_vector_to_half(__half* out, const float* ptr, size_t n);

void half_matrix_to_float(float* out, const __half* ptr, size_t m, size_t n, size_t pitch);

void float_matrix_to_half(__half* out, const float* ptr, size_t m, size_t n, size_t pitch);

}  // namespace cunumeric
