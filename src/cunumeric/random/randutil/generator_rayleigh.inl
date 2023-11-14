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
struct rayleigh_t;

template <>
struct rayleigh_t<float> {
  float sigma;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS float operator()(gen_t& gen)
  {
#ifdef USE_STL_RANDOM_ENGINE_
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);  // [0, 1)
    auto y = dis(gen);

    // bring to (0, 1]; y cannot be 0:
    y = 1 - y;
#else
    auto y = curand_uniform(&gen);  // returns (0, 1]; y cannot be 0
#endif
    return sigma * ::sqrtf(-2.0f * ::logf(y));
  }
};

template <>
struct rayleigh_t<double> {
  double sigma;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS double operator()(gen_t& gen)
  {
#ifdef USE_STL_RANDOM_ENGINE_
    std::uniform_real_distribution<double> dis(0.0, 1.0);  // [0, 1)
    auto y = dis(gen);

    // bring to (0, 1]; y cannot be 0:
    y = 1 - y;
#else
    auto y = curand_uniform_double(&gen);  // returns (0, 1]; y cannot be 0
#endif
    return sigma * ::sqrt(-2.0 * ::log(y));
  }
};
