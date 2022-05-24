// (c) 2022 NVIDIA all rights reserved
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
