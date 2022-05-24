// (c) 2022 NVIDIA all rights reserved
#include "generator.cuh"

template <typename field_t>
struct raw;

template <>
struct raw<uint32_t> {
  template <typename gen_t>
  __forceinline__ __host__ __device__ uint32_t operator()(gen_t& gen)
  {
    return (uint32_t)curand(&gen);
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateRawUInt32Ex(curandGeneratorEx_t generator,
                                                              uint32_t* outputPtr,
                                                              size_t n)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  raw<uint32_t> func;
  return curandimpl::dispatch_sample<raw<uint32_t>, uint32_t>(gen, func, n, outputPtr);
}
