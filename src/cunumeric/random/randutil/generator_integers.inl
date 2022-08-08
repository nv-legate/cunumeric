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

#include "generator.h"

template <typename field_t>
struct integers;

template <>
struct integers<int16_t> {
  int16_t from;
  int16_t to;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS int32_t operator()(gen_t& gen)
  {
    return (int16_t)(curand(&gen) % (uint16_t)(to - from)) + from;
  }
};

template <>
struct integers<int32_t> {
  int32_t from;
  int32_t to;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS int32_t operator()(gen_t& gen)
  {
    return (int32_t)(curand(&gen) % (uint32_t)(to - from)) + from;
  }
};

template <>
struct integers<int64_t> {
  int64_t from;
  int64_t to;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS int64_t operator()(gen_t& gen)
  {
    // take two draws to get a 64 bits value
    unsigned low  = curand(&gen);
    unsigned high = curand(&gen);
    return (int64_t)((((uint64_t)high << 32) | (uint64_t)low) % (to - from)) + from;
  }
};
