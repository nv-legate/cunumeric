/* Copyright 2023 NVIDIA Corporation
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

// attempt to masquerade as MacOS on host:
//
// #ifndef LEGATE_USE_CUDA
// #define USE_STL_RANDOM_ENGINE_
// #endif

#ifdef USE_STL_RANDOM_ENGINE_

// #pragma message("************ STL path *************")

#include <random>

using rnd_status_t = int;
enum class randRngType : int { STL_MT_19937 = 1 };
using randRngType_t              = randRngType;
constexpr int RND_STATUS_SUCCESS = 0;

// same as for curand:
//
constexpr rnd_status_t RND_STATUS_INTERNAL_ERROR = 999;
constexpr rnd_status_t RND_STATUS_TYPE_ERROR     = 103;

namespace randutilimpl {
constexpr int RND_RNG_PSEUDO_XORWOW        = randRngType::STL_MT_19937;
constexpr int RND_RNG_PSEUDO_PHILOX4_32_10 = randRngType::STL_MT_19937;
constexpr int RND_RNG_PSEUDO_MRG32K3A      = randRngType::STL_MT_19937;

using gen_XORWOW_t        = std::mt19937;
using gen_Philox4_32_10_t = std::mt19937;
using gen_MRG32k3a_t      = std::mt19937;
}  // namespace randutilimpl

using stream_t = void*;
#else
#include <curand_kernel.h>

// #pragma message("************ CURAND path ************")

using rnd_status_t                               = curandStatus_t;
using randRngType                                = curandRngType;
using randRngType_t                              = curandRngType_t;
constexpr rnd_status_t RND_STATUS_SUCCESS        = CURAND_STATUS_SUCCESS;
constexpr rnd_status_t RND_STATUS_INTERNAL_ERROR = CURAND_STATUS_INTERNAL_ERROR;
constexpr rnd_status_t RND_STATUS_TYPE_ERROR     = CURAND_STATUS_TYPE_ERROR;

constexpr int RND_RNG_PSEUDO_XORWOW        = CURAND_RNG_PSEUDO_XORWOW;
constexpr int RND_RNG_PSEUDO_PHILOX4_32_10 = CURAND_RNG_PSEUDO_PHILOX4_32_10;
constexpr int RND_RNG_PSEUDO_MRG32K3A      = CURAND_RNG_PSEUDO_MRG32K3A;

using gen_XORWOW_t        = curandStateXORWOW_t;
using gen_Philox4_32_10_t = curandStatePhilox4_32_10_t;
using gen_MRG32k3a_t      = curandStateMRG32k3a_t;

using stream_t = cudaStream_t;

#endif
