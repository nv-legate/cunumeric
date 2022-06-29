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
#include "generator_create.inl"

extern "C" curandStatus_t CURANDAPI randutilCreateGeneratorHost(randutilGenerator_t* generator,
                                                                curandRngType_t rng_type,
                                                                uint64_t seed,
                                                                uint64_t generatorID,
                                                                uint32_t flags)
{
  return inner_randutilCreateGenerator<randutilimpl::execlocation::HOST>(
    generator, rng_type, seed, generatorID, nullptr);
}

extern "C" curandStatus_t CURANDAPI randutilDestroyGenerator(randutilGenerator_t generator)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  try {
    gen->destroy();
    delete gen;

    return CURAND_STATUS_SUCCESS;
  } catch (int errorCode) {
    delete gen;
    return (curandStatus_t)errorCode;
  }
}

namespace randutilimpl {

// HOST-side template instantiation of generator
template <typename func_t, typename out_t>
struct dispatcher<randutilimpl::execlocation::HOST, func_t, out_t> {
  static curandStatus_t run(randutilimpl::basegenerator* gen, func_t func, size_t N, out_t* out)
  {
    return inner_dispatch_sample<randutilimpl::execlocation::HOST, func_t, out_t>(
      gen, func, N, out);
  }
};

template <typename func_t, typename out_t>
curandStatus_t dispatch(randutilimpl::basegenerator* gen, func_t func, size_t N, out_t* out)
{
  switch (gen->location()) {
    case randutilimpl::execlocation::HOST:
      return dispatcher<randutilimpl::execlocation::HOST, func_t, out_t>::run(gen, func, N, out);
    case randutilimpl::execlocation::DEVICE:
      return dispatcher<randutilimpl::execlocation::DEVICE, func_t, out_t>::run(gen, func, N, out);
    default: return CURAND_STATUS_INTERNAL_ERROR;
  }
}

}  // namespace randutilimpl

#pragma region integers

#include "generator_integers.inl"

extern "C" curandStatus_t CURANDAPI randutilGenerateIntegers32(
  randutilGenerator_t generator, int32_t* outputPtr, size_t n, int32_t low, int32_t high)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  integers<int32_t> func;
  func.from = low;
  func.to   = high;
  return randutilimpl::dispatch<decltype(func), int32_t>(gen, func, n, outputPtr);
}

extern "C" curandStatus_t CURANDAPI randutilGenerateIntegers64(
  randutilGenerator_t generator, int64_t* outputPtr, size_t n, int64_t low, int64_t high)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  integers<int64_t> func;
  func.from = low;
  func.to   = high;
  return randutilimpl::dispatch<decltype(func), int64_t>(gen, func, n, outputPtr);
}

#pragma endregion