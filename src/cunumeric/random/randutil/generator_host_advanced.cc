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

#pragma region beta

#include "generator_beta.inl"

extern "C" curandStatus_t randutilGenerateBetaEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float a, float b)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  beta_t<float> func;
  func.a = a;
  func.b = b;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" curandStatus_t randutilGenerateBetaDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double a, double b)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  beta_t<double> func;
  func.a = a;
  func.b = b;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region FisherSnedecor

#include "generator_f.inl"

extern "C" curandStatus_t randutilGenerateFisherSnedecorEx(randutilGenerator_t generator,
                                                           float* outputPtr,
                                                           size_t n,
                                                           float dfnum,
                                                           float dfden,
                                                           float nonc)  // 0.0f is F distribution
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  if (nonc == 0.0f) {
    f_t<float> func;
    func.dfnum = dfnum;
    func.dfden = dfden;
    return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
  } else {
    noncentralf_t<float> func;
    func.dfnum = dfnum;
    func.dfden = dfden;
    func.nonc  = nonc;
    return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
  }
}

extern "C" curandStatus_t randutilGenerateFisherSnedecorDoubleEx(
  randutilGenerator_t generator,
  double* outputPtr,
  size_t n,
  double dfnum,
  double dfden,
  double nonc)  // 0.0 is F distribution
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  if (nonc == 0.0) {
    f_t<double> func;
    func.dfnum = dfnum;
    func.dfden = dfden;
    return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
  } else {
    noncentralf_t<double> func;
    func.dfnum = dfnum;
    func.dfden = dfden;
    func.nonc  = nonc;
    return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
  }
}

#pragma endregion

#pragma region logseries

#include "generator_logseries.inl"

extern "C" curandStatus_t randutilGenerateLogSeriesEx(randutilGenerator_t generator,
                                                      uint32_t* outputPtr,
                                                      size_t n,
                                                      double p)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  logseries_t<double> func;
  func.p = p;
  return randutilimpl::dispatch<decltype(func), uint32_t>(gen, func, n, outputPtr);
}

#pragma endregion
