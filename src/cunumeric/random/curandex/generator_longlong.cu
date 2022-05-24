// (c) 2022 NVIDIA all rights reserved
#include "generator.cuh"

struct longlong {
  template <typename gen_t>
  __forceinline__ __host__ __device__ uint64_t operator()(gen_t& gen)
  {
    // take two draws to get a 64 bits value
    unsigned low  = curand(&gen);
    unsigned high = curand(&gen);
    return ((uint64_t)high << 32) | (uint64_t)low;
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateLongLongEx(curandGeneratorEx_t generator,
                                                             uint64_t* outputPtr,
                                                             size_t n)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  longlong func;
  return curandimpl::dispatch_sample<longlong, uint64_t>(gen, func, n, outputPtr);
}
