// (c) 2022 NVIDIA all rights reserved
#include "generator.cuh"

template <typename field_t>
struct power_t;

template <>
struct power_t<float> {
  float xm, invalpha;

  template <typename gen_t>
  __forceinline__ __host__ __device__ float operator()(gen_t& gen)
  {
    float y = curand_uniform(&gen);  // y cannot be 0
    return ::expf(::logf(y) * invalpha);
  }
};

extern "C" curandStatus_t CURANDAPI curandGeneratePowerEx(curandGeneratorEx_t generator,
                                                          float* outputPtr,
                                                          size_t num,
                                                          float alpha)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  power_t<float> func;
  func.invalpha = 1.0f / alpha;
  return curandimpl::dispatch_sample<power_t<float>, float>(gen, func, num, outputPtr);
}

template <>
struct power_t<double> {
  double xm, invalpha;

  template <typename gen_t>
  __forceinline__ __host__ __device__ double operator()(gen_t& gen)
  {
    double y = curand_uniform_double(&gen);  // y cannot be 0 -- use y as 1-cdf(x)
    return ::exp(::log(y) * invalpha);
  }
};

extern "C" curandStatus_t CURANDAPI curandGeneratePowerDoubleEx(curandGeneratorEx_t generator,
                                                                double* outputPtr,
                                                                size_t num,
                                                                double alpha)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  power_t<double> func;
  func.invalpha = 1.0 / alpha;
  return curandimpl::dispatch_sample<power_t<double>, double>(gen, func, num, outputPtr);
}