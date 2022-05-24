// (c) 2022 NVIDIA all rights reserved
#include "generator.cuh"

template <typename field_t>
struct rayleigh_t;

template <>
struct rayleigh_t<float> {
  float sigma;

  template <typename gen_t>
  __forceinline__ __host__ __device__ float operator()(gen_t& gen)
  {
    float y = curand_uniform(&gen);  // y cannot be 0
    return sigma * ::sqrtf(-2.0f * ::logf(y));
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateRayleighEx(curandGeneratorEx_t generator,
                                                             float* outputPtr,
                                                             size_t num,
                                                             float sigma)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  rayleigh_t<float> func;
  func.sigma = sigma;
  return curandimpl::dispatch_sample<rayleigh_t<float>, float>(gen, func, num, outputPtr);
}

template <>
struct rayleigh_t<double> {
  double sigma;

  template <typename gen_t>
  __forceinline__ __host__ __device__ double operator()(gen_t& gen)
  {
    double y = curand_uniform_double(&gen);  // y cannot be 0
    return sigma * ::sqrt(-2.0 * ::log(y));
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateRayleighDoubleEx(curandGeneratorEx_t generator,
                                                                   double* outputPtr,
                                                                   size_t num,
                                                                   double sigma)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  rayleigh_t<double> func;
  func.sigma = sigma;
  return curandimpl::dispatch_sample<rayleigh_t<double>, double>(gen, func, num, outputPtr);
}
