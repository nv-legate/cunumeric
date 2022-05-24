// (c) 2022 NVIDIA all rights reserved
#include "generator.cuh"

template <typename field_t>
struct gumbel_t;

template <>
struct gumbel_t<float> {
  float mu, beta;

  // gumble cdf : $ cdf(x) = \exp^{-\exp^{-\frac{x-\mu}{\beta}}} $
  template <typename gen_t>
  __forceinline__ __host__ __device__ float operator()(gen_t& gen)
  {
    float y = curand_uniform(&gen);  // y cannot be zero
    if (y == 1.0f) return mu;
    float lny = ::logf(y);
    return mu - beta * ::logf(-lny);
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateGumbelEx(
  curandGeneratorEx_t generator, float* outputPtr, size_t num, float mu, float beta)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  gumbel_t<float> func;
  func.mu   = mu;
  func.beta = beta;
  return curandimpl::dispatch_sample<gumbel_t<float>, float>(gen, func, num, outputPtr);
}

template <>
struct gumbel_t<double> {
  double mu, beta;

  template <typename gen_t>
  __forceinline__ __host__ __device__ double operator()(gen_t& gen)
  {
    double y = curand_uniform_double(&gen);  // y cannot be zero
    if (y == 1.0) return mu;
    double lny = ::log(y);
    return mu - beta * ::log(-lny);
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateGumbelDoubleEx(
  curandGeneratorEx_t generator, double* outputPtr, size_t num, double mu, double beta)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  gumbel_t<double> func;
  func.mu   = mu;
  func.beta = beta;
  return curandimpl::dispatch_sample<gumbel_t<double>, double>(gen, func, num, outputPtr);
}