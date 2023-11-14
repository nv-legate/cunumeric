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
struct gumbel_t;

template <>
struct gumbel_t<float> {
  float mu, beta;

  // gumble cdf : $ cdf(x) = \exp^{-\exp^{-\frac{x-\mu}{\beta}}} $
  template <typename gen_t>
  RANDUTIL_QUALIFIERS float operator()(gen_t& gen)
  {
#ifdef USE_STL_RANDOM_ENGINE_
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    auto y = dis(gen);  // returns [0, 1);

    // bring to (0, 1]:
    y = 1 - y;
#else
    float y = curand_uniform(&gen);  // y cannot be zero
#endif

    if (y == 1.0f) return mu;
    float lny = ::logf(y);
    return mu - beta * ::logf(-lny);
  }
};

template <>
struct gumbel_t<double> {
  double mu, beta;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS double operator()(gen_t& gen)
  {
#ifdef USE_STL_RANDOM_ENGINE_
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto y = dis(gen);  // returns [0, 1);

    // bring to (0, 1]:
    y = 1 - y;
#else
    double y = curand_uniform_double(&gen);  // y cannot be zero
#endif

    if (y == 1.0) return mu;
    double lny = ::log(y);
    return mu - beta * ::log(-lny);
  }
};
