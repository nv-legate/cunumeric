// (c) 2022 NVIDIA all rights reserved
#include "generator.cuh"

template <typename field_t>
struct weilbull_t;

template <>
struct weilbull_t<float> {
  float lambda, invk;

  template <typename gen_t>
  __forceinline__ __host__ __device__ float operator()(gen_t& gen)
  {
    float y = curand_uniform(&gen);  // y cannot be 0
    // log(y) can be zero !
    float lny = ::logf(y);
    if (lny == 0.0f) return 0.0f;
    return lambda * ::expf(::logf(-lny) * invk);
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateWeibullEx(
  curandGeneratorEx_t generator, float* outputPtr, size_t num, float lambda, float k)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  weilbull_t<float> func;
  func.lambda = lambda;
  func.invk   = 1.0f / k;
  return curandimpl::dispatch_sample<weilbull_t<float>, float>(gen, func, num, outputPtr);
}

template <>
struct weilbull_t<double> {
  double lambda, invk;

  template <typename gen_t>
  __forceinline__ __host__ __device__ double operator()(gen_t& gen)
  {
    double y = curand_uniform_double(&gen);  // y cannot be 0
    // log(y) can be zero !
    float lny = ::log(y);
    if (lny == 0.0f) return 0.0f;
    return lambda * ::exp(::log(-lny) * invk);
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateWeibullDoubleEx(
  curandGeneratorEx_t generator, double* outputPtr, size_t num, double lambda, double k)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  weilbull_t<double> func;
  func.lambda = lambda;
  func.invk   = 1.0 / k;
  return curandimpl::dispatch_sample<weilbull_t<double>, double>(gen, func, num, outputPtr);
}
