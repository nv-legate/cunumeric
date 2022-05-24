// (c) 2022 NVIDIA all rights reserved
#include "generator.cuh"

template <typename field_t>
struct uniform_t;

template <>
struct uniform_t<float> {
  float offset, mult;

  template <typename gen_t>
  __forceinline__ __host__ __device__ float operator()(gen_t& gen)
  {
    return offset + mult * curand_uniform(&gen);
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateUniformEx(
  curandGeneratorEx_t generator, float* outputPtr, size_t n, float low, float high)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  uniform_t<float> func;
  func.offset = low;
  func.mult   = high - low;
  return curandimpl::dispatch_sample<uniform_t<float>, float>(gen, func, n, outputPtr);
}

template <>
struct uniform_t<double> {
  double offset, mult;

  template <typename gen_t>
  __forceinline__ __host__ __device__ double operator()(gen_t& gen)
  {
    return offset + mult * curand_uniform_double(&gen);
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateUniformDoubleEx(
  curandGeneratorEx_t generator, double* outputPtr, size_t n, double low, double high)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  uniform_t<double> func;
  func.offset = low;
  func.mult   = high - low;
  return curandimpl::dispatch_sample<uniform_t<double>, double>(gen, func, n, outputPtr);
}