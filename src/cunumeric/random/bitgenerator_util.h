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

// Match these to BitGeneratorOperation in config.py
enum class BitGeneratorOperation : int32_t {
  CREATE       = CUNUMERIC_BITGENOP_CREATE,
  DESTROY      = CUNUMERIC_BITGENOP_DESTROY,
  RAND_RAW     = CUNUMERIC_BITGENOP_RAND_RAW,
  DISTRIBUTION = CUNUMERIC_BITGENOP_DISTRIBUTION,
};

// Match these to BitGeneratorType in config.py
enum class BitGeneratorType : uint32_t {
  DEFAULT       = CUNUMERIC_BITGENTYPE_DEFAULT,
  XORWOW        = CUNUMERIC_BITGENTYPE_XORWOW,
  MRG32K3A      = CUNUMERIC_BITGENTYPE_MRG32K3A,
  MTGP32        = CUNUMERIC_BITGENTYPE_MTGP32,
  MT19937       = CUNUMERIC_BITGENTYPE_MT19937,
  PHILOX4_32_10 = CUNUMERIC_BITGENTYPE_PHILOX4_32_10,
};

// Match these to BitGeneratorDistribution in config.py
enum class BitGeneratorDistribution : int32_t {
  INTEGERS_32 = CUNUMERIC_BITGENDIST_INTEGERS_32,
  INTEGERS_64 = CUNUMERIC_BITGENDIST_INTEGERS_64,
};

}  // namespace cunumeric
