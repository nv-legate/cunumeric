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

#pragma region exponential

#include "generator_exponential.inl"

extern "C" curandStatus_t CURANDAPI randutilGenerateExponentialEx(randutilGenerator_t generator,
                                                                  float* outputPtr,
                                                                  size_t n,
                                                                  float scale)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  exponential_t<float> func;
  func.scale = scale;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" curandStatus_t CURANDAPI randutilGenerateExponentialDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double scale)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  exponential_t<double> func;
  func.scale = scale;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region gumbel

#include "generator_gumbel.inl"

extern "C" curandStatus_t CURANDAPI randutilGenerateGumbelEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float beta)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  gumbel_t<float> func;
  func.mu   = mu;
  func.beta = beta;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" curandStatus_t CURANDAPI randutilGenerateGumbelDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double beta)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  gumbel_t<double> func;
  func.mu   = mu;
  func.beta = beta;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region laplace

#include "generator_laplace.inl"

extern "C" curandStatus_t CURANDAPI randutilGenerateLaplaceEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float beta)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  laplace_t<float> func;
  func.mu   = mu;
  func.beta = beta;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" curandStatus_t CURANDAPI randutilGenerateLaplaceDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double beta)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  laplace_t<double> func;
  func.mu   = mu;
  func.beta = beta;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion
