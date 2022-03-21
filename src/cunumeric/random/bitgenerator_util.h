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

#include "cunumeric/cunumeric.h"

namespace cunumeric {

enum class BitGeneratorOperation : int32_t {
    CREATE = 1,
    DESTROY = 2,
};

enum class BitGeneratorType : int32_t {
    DEFAULT = 0,
    XORWOW = 1,
    MRG32K3A = 2,
    MTGP32 = 3,
    MT19937 = 4,
    PHILOX4_32_10 = 5,
};

}  // namespace cunumeric