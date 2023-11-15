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

#ifndef LEGATE_USE_CUDA

// assume on MACOS for testing:
//
#define IS_MAC_OS_ 1

#if IS_MAC_OS_ == 1
#define USE_STL_RANDOM_ENGINE_
#endif

#endif

#include "generator.h"

#include "randomizer.h"

template <typename field_t>
struct uniform_t;

template <>
struct uniform_t<float> {
  float offset, mult;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS float operator()(gen_t& gen)
  {
    auto y = randutilimpl::engine_uniform_single(gen);  // returns (0, 1];
    return offset + mult * y;
  }
};

template <>
struct uniform_t<double> {
  double offset, mult;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS double operator()(gen_t& gen)
  {
    auto y = randutilimpl::engine_uniform_double(gen);  // returns (0, 1];
    return offset + mult * y;
  }
};
