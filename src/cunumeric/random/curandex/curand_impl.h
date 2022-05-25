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
#include "curand_ex.h"

#define CURAND_CHECK_LINE(a, file, line)                          \
  {                                                               \
    curandStatus_t __curer = a;                                   \
    if (CURAND_STATUS_SUCCESS != __curer) { throw(int) __curer; } \
  }
#define CURAND_CHECK(a) CURAND_CHECK_LINE(a, __FILE__, __LINE__)

#define CU_CHECK_LINE(a, file, line)                \
  {                                                 \
    CUresult __cuer = a;                            \
    if (CUDA_SUCCESS != __cuer) {                   \
      if (__cuer == CUDA_ERROR_OUT_OF_MEMORY)       \
        throw(int) CURAND_STATUS_ALLOCATION_FAILED; \
      else                                          \
        throw(int) CURAND_STATUS_INTERNAL_ERROR;    \
    }                                               \
  }
#define CU_CHECK(a) CU_CHECK_LINE(a, __FILE__, __LINE__)

#define CUDA_CHECK_LINE(a, file, line)              \
  {                                                 \
    cudaError_t __cuer = a;                         \
    if (cudaSuccess != __cuer) {                    \
      if (__cuer == cudaErrorMemoryAllocation)      \
        throw(int) CURAND_STATUS_ALLOCATION_FAILED; \
      else                                          \
        throw(int) CURAND_STATUS_INTERNAL_ERROR;    \
    }                                               \
  }
#define CUDA_CHECK(a) CUDA_CHECK_LINE(a, __FILE__, __LINE__)

namespace curandimpl {

enum class execlocation : int { DEVICE = 0, HOST = 1 };

template <typename gen_t, execlocation loc>
struct inner_generator;

}  // namespace curandimpl