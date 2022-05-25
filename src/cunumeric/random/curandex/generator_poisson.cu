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

#include "generator.cuh"

struct poisson {
  double lambda = 1.0;

  template <typename gen_t>
  __forceinline__ __host__ __device__ unsigned operator()(gen_t& gen)
  {
    return curand_poisson(&gen, lambda);
  }
};

extern "C" curandStatus_t CURANDAPI curandGeneratePoissonEx(curandGeneratorEx_t generator,
                                                            uint32_t* outputPtr,
                                                            size_t n,
                                                            double lambda)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  poisson func;
  func.lambda = lambda;
  return curandimpl::dispatch_sample<poisson, uint32_t>(gen, func, n, outputPtr);
}
