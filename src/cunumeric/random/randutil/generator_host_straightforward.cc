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

#pragma region logistic

#include "generator_logistic.inl"

extern "C" curandStatus_t CURANDAPI randutilGenerateLogisticEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float beta)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  logistic_t<float> func;
  func.mu   = mu;
  func.beta = beta;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" curandStatus_t CURANDAPI randutilGenerateLogisticDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double beta)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  logistic_t<double> func;
  func.mu   = mu;
  func.beta = beta;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region pareto

#include "generator_pareto.inl"

extern "C" curandStatus_t CURANDAPI randutilGenerateParetoEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float xm, float alpha)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  pareto_t<float> func;
  func.xm       = xm;
  func.invalpha = 1.0f / alpha;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" curandStatus_t CURANDAPI randutilGenerateParetoDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double xm, double alpha)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  pareto_t<double> func;
  func.xm       = xm;
  func.invalpha = 1.0 / alpha;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region power

#include "generator_power.inl"

extern "C" curandStatus_t CURANDAPI randutilGeneratePowerEx(randutilGenerator_t generator,
                                                            float* outputPtr,
                                                            size_t n,
                                                            float alpha)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  power_t<float> func;
  func.invalpha = 1.0f / alpha;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" curandStatus_t CURANDAPI randutilGeneratePowerDoubleEx(randutilGenerator_t generator,
                                                                  double* outputPtr,
                                                                  size_t n,
                                                                  double alpha)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  power_t<double> func;
  func.invalpha = 1.0 / alpha;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region rayleigh

#include "generator_rayleigh.inl"

extern "C" curandStatus_t CURANDAPI randutilGenerateRayleighEx(randutilGenerator_t generator,
                                                               float* outputPtr,
                                                               size_t n,
                                                               float sigma)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  rayleigh_t<float> func;
  func.sigma = sigma;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" curandStatus_t CURANDAPI randutilGenerateRayleighDoubleEx(randutilGenerator_t generator,
                                                                     double* outputPtr,
                                                                     size_t n,
                                                                     double sigma)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  rayleigh_t<double> func;
  func.sigma = sigma;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region cauchy

#include "generator_cauchy.inl"

extern "C" curandStatus_t CURANDAPI randutilGenerateCauchyEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float x0, float gamma)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  cauchy_t<float> func;
  func.x0    = x0;
  func.gamma = gamma;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" curandStatus_t CURANDAPI randutilGenerateCauchyDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double x0, double gamma)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  cauchy_t<double> func;
  func.x0    = x0;
  func.gamma = gamma;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion
