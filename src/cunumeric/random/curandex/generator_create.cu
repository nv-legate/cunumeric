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

template <typename gen_t, curandimpl::execlocation location>
curandStatus_t createGeneratorEx(curandGeneratorEx_t* generator,
                                 uint64_t seed,
                                 uint64_t generatorID,
                                 cudaStream_t stream = nullptr)
{
  try {
    curandimpl::inner_generator<gen_t, location>* result =
      new curandimpl::inner_generator<gen_t, location>(seed, generatorID, stream);
    *generator = (curandGeneratorEx_t)result;
    return CURAND_STATUS_SUCCESS;
  } catch (int errorCode) {
    return (curandStatus_t)errorCode;
  }
}

template <curandimpl::execlocation location>
static curandStatus_t CURANDAPI inner_curandCreateGeneratorEx(curandGeneratorEx_t* generator,
                                                              curandRngType_t rng_type,
                                                              uint64_t seed,
                                                              uint64_t generatorID,
                                                              cudaStream_t stream = nullptr)
{
  switch (rng_type) {
    case CURAND_RNG_PSEUDO_XORWOW:
      return createGeneratorEx<curandStateXORWOW_t, location>(generator, seed, generatorID, stream);
    case CURAND_RNG_PSEUDO_PHILOX4_32_10:
      return createGeneratorEx<curandStatePhilox4_32_10_t, location>(
        generator, seed, generatorID, stream);
    case CURAND_RNG_PSEUDO_MRG32K3A:
      return createGeneratorEx<curandStateMRG32k3a_t, location>(
        generator, seed, generatorID, stream);
    default: return CURAND_STATUS_TYPE_ERROR;
  }
}

extern "C" curandStatus_t CURANDAPI curandCreateGeneratorEx(curandGeneratorEx_t* generator,
                                                            curandRngType_t rng_type,
                                                            uint64_t seed,
                                                            uint64_t generatorID,
                                                            uint32_t flags,
                                                            cudaStream_t stream)
{
  return inner_curandCreateGeneratorEx<curandimpl::execlocation::DEVICE>(
    generator, rng_type, seed, generatorID, stream);
}

extern "C" curandStatus_t CURANDAPI curandCreateGeneratorHostEx(curandGeneratorEx_t* generator,
                                                                curandRngType_t rng_type,
                                                                uint64_t seed,
                                                                uint64_t generatorID,
                                                                uint32_t flags)
{
  return inner_curandCreateGeneratorEx<curandimpl::execlocation::HOST>(
    generator, rng_type, seed, generatorID, nullptr);
}

extern "C" curandStatus_t CURANDAPI curandDestroyGeneratorEx(curandGeneratorEx_t generator)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  try {
    gen->destroy();
    delete gen;

    return CURAND_STATUS_SUCCESS;
  } catch (int errorCode) {
    delete gen;
    return (curandStatus_t)errorCode;
  }
}