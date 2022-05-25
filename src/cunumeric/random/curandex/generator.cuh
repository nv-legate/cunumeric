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

#pragma once

// also allow usage of generators on host
#define QUALIFIERS static __forceinline__ __device__ __host__

#include <cstdint>
#include <cassert>
#include "curand_impl.h"

#include <curand_kernel.h>

namespace curandimpl {
static constexpr int blocksPerMultiProcessor = 2;    // TODO: refine => number of blocks per mp
static constexpr int blockDimX               = 256;  // TODO: refine ?

extern __shared__ char local_shared[];

template <typename gen_t>
__global__ void __launch_bounds__(blockDimX, blocksPerMultiProcessor)
  initgenerators(gen_t* gens, uint64_t seed, uint64_t generatorID)
{
  static_assert((sizeof(gen_t) & 3) == 0, "alignment issues");
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  // TODO: preload to shared
  gen_t gen = gens[id];
  // get offset from ID
  unsigned baseOffset = __brev((unsigned)id);
  uint64_t offset     = ((uint64_t)baseOffset) << 32;
  // init the generator
  curand_init(seed, generatorID, offset, &gen);
  // store back
  gens[id] = gen;
}

template <typename gen_t, typename func_t, typename out_t>
__global__ void __launch_bounds__(blockDimX, blocksPerMultiProcessor)
  gpu_draw(int ngenerators, gen_t* gens, func_t func, size_t N, out_t* draws)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  assert(id < ngenerators);
  // TODO: improve load
  gen_t gen = gens[id];
  for (size_t k = id; k < N; k += blockDim.x * gridDim.x) { draws[k] = func(gen); }
  // save state
  gens[id] = gen;
}

struct basegenerator {
  virtual int generatorTypeId()   = 0;
  virtual execlocation location() = 0;
  virtual void destroy()          = 0;  // avoid exceptions in destructor
  virtual ~basegenerator() {}
};

template <typename gen_t>
struct generatorid;

template <>
struct generatorid<curandStateXORWOW_t> {
  static constexpr int rng_type = CURAND_RNG_PSEUDO_XORWOW;
};
template <>
struct generatorid<curandStatePhilox4_32_10_t> {
  static constexpr int rng_type = CURAND_RNG_PSEUDO_PHILOX4_32_10;
};
template <>
struct generatorid<curandStateMRG32k3a_t> {
  static constexpr int rng_type = CURAND_RNG_PSEUDO_MRG32K3A;
};

template <int gen_id>
struct generatortype;

template <>
struct generatortype<CURAND_RNG_PSEUDO_XORWOW> {
  using rng_t = curandStateXORWOW_t;
};
template <>
struct generatortype<CURAND_RNG_PSEUDO_PHILOX4_32_10> {
  using rng_t = curandStatePhilox4_32_10_t;
};
template <>
struct generatortype<CURAND_RNG_PSEUDO_MRG32K3A> {
  using rng_t = curandStateMRG32k3a_t;
};

template <typename gen_t>
struct inner_generator<gen_t, curandimpl::execlocation::DEVICE> : basegenerator {
  uint64_t seed;
  uint64_t generatorID;
  cudaStream_t stream;

  int multiProcessorCount = 0;
  bool asyncsupported     = true;
  int ngenerators;
  gen_t* generators = nullptr;

  inner_generator(uint64_t seed, uint64_t generatorID, cudaStream_t stream)
    : seed(seed), generatorID(generatorID), stream(stream)
  {
    int deviceId;
    CUDA_CHECK(::cudaGetDevice(&deviceId));
    CU_CHECK(::cuDeviceGetAttribute(
      &multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, deviceId));
    // get number of generators
    ngenerators = blockDimX * multiProcessorCount * blocksPerMultiProcessor;
    if (ngenerators == 0) throw(int) CURAND_STATUS_INTERNAL_ERROR;

    // allocate buffer for generators state
    int driverVersion, runtimeVersion;
    CUDA_CHECK(::cudaDriverGetVersion(&driverVersion));
    if (driverVersion < 10020) asyncsupported = false;
    CUDA_CHECK(::cudaRuntimeGetVersion(&runtimeVersion));
    if (runtimeVersion < 10020) asyncsupported = false;
    if (asyncsupported) {
#if (__CUDACC_VER_MAJOR__ > 11 || ((__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 2)))
      CUDA_CHECK(::cudaMallocAsync(&generators, ngenerators * sizeof(gen_t), stream));
#else
      CUDA_CHECK(::cudaMalloc(&generators, ngenerators * sizeof(gen_t)));
#endif
    } else
      CUDA_CHECK(::cudaMalloc(&generators, ngenerators * sizeof(gen_t)));

    // initialize generators
    initgenerators<<<blocksPerMultiProcessor * multiProcessorCount, blockDimX, 0, stream>>>(
      generators, seed, generatorID);
    CUDA_CHECK(::cudaPeekAtLastError());
  }

  virtual void destroy() override
  {
    CUDA_CHECK(::cudaStreamSynchronize(stream));
    if (asyncsupported) {
#if (__CUDACC_VER_MAJOR__ > 11 || ((__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 2)))
      CUDA_CHECK(::cudaFreeAsync(generators, stream));
#else
      CUDA_CHECK(::cudaFree(generators));
#endif
    } else
      CUDA_CHECK(::cudaFree(generators));

    generators = nullptr;
  }

  virtual execlocation location() override { return curandimpl::execlocation::DEVICE; }

  virtual int generatorTypeId() override { return generatorid<gen_t>::rng_type; }

  virtual ~inner_generator()
  {
    // TODO: warning message if generator has not been destroyed
  }

  template <typename func_t, typename out_t>
  curandStatus_t draw(func_t func, size_t N, out_t* out)
  {
    if (generators == nullptr)  // destroyed was called
      return CURAND_STATUS_NOT_INITIALIZED;
    gpu_draw<gen_t, func_t, out_t><<<multiProcessorCount * blocksPerMultiProcessor, blockDimX>>>(
      ngenerators, generators, func, N, out);
    return ::cudaPeekAtLastError() == cudaSuccess ? CURAND_STATUS_SUCCESS
                                                  : CURAND_STATUS_INTERNAL_ERROR;
  }
};

template <typename gen_t>
struct inner_generator<gen_t, curandimpl::execlocation::HOST> : basegenerator {
  uint64_t seed;
  uint64_t generatorID;
  gen_t generator;

  inner_generator(uint64_t seed, uint64_t generatorID, cudaStream_t ignored)
    : seed(seed), generatorID(generatorID)
  {
    curand_init(seed, generatorID, 0, &generator);
  }

  virtual void destroy() override {}

  virtual execlocation location() override { return curandimpl::execlocation::HOST; }

  virtual int generatorTypeId() override { return generatorid<gen_t>::rng_type; }

  virtual ~inner_generator() {}

  template <typename func_t, typename out_t>
  curandStatus_t draw(func_t func, size_t N, out_t* out)
  {
    for (size_t k = 0; k < N; ++k) { out[k] = func(generator); }
    return CURAND_STATUS_SUCCESS;
  }
};

template <curandimpl::execlocation location, typename func_t, typename out_t>
curandStatus_t inner_dispatch_sample(basegenerator* gen, func_t func, size_t N, out_t* out)
{
  switch (gen->generatorTypeId()) {
    case CURAND_RNG_PSEUDO_XORWOW:
      return static_cast<inner_generator<curandStateXORWOW_t, location>*>(gen)
        ->template draw<func_t, out_t>(func, N, out);
    case CURAND_RNG_PSEUDO_PHILOX4_32_10:
      return static_cast<inner_generator<curandStatePhilox4_32_10_t, location>*>(gen)
        ->template draw<func_t, out_t>(func, N, out);
    case CURAND_RNG_PSEUDO_MRG32K3A:
      return static_cast<inner_generator<curandStateMRG32k3a_t, location>*>(gen)
        ->template draw<func_t, out_t>(func, N, out);
    default: return CURAND_STATUS_INTERNAL_ERROR;
  }
}

template <typename func_t, typename out_t>
curandStatus_t dispatch_sample(basegenerator* gen, func_t func, size_t N, out_t* out)
{
  switch (gen->location()) {
    case curandimpl::execlocation::HOST:
      return inner_dispatch_sample<curandimpl::execlocation::HOST, func_t, out_t>(
        gen, func, N, out);
    case curandimpl::execlocation::DEVICE:
      return inner_dispatch_sample<curandimpl::execlocation::DEVICE, func_t, out_t>(
        gen, func, N, out);
    default: return CURAND_STATUS_INTERNAL_ERROR;
  }
}

}  // namespace curandimpl