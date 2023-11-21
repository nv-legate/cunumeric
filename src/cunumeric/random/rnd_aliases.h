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

#ifdef USE_STL_RANDOM_ENGINE_
using rnd_status_t = int;
enum class randRngType : int { STL_MT19937 = 1 };
using randRngType_t              = randRngType;
constexpr int RND_STATUS_SUCCESS = 0;
#else
using rnd_status_t                        = curandStatus_t;
using randRngType                         = curandRngType;
using randRngType_t                       = curandRngType_t;
constexpr rnd_status_t RND_STATUS_SUCCESS = CURAND_STATUS_SUCCESS;
#endif
