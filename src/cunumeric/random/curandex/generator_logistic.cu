// (c) 2022 NVIDIA all rights reserved
#include "generator.cuh"

template <typename field_t>
struct logistic_t;

template <>
struct logistic_t<float> {
  float mu, beta;

  template <typename gen_t>
  __forceinline__ __host__ __device__ float operator()(gen_t& gen)
  {
    float y = curand_uniform(&gen);  // y cannot be 0
    float t = 1.0f / y - 1.0f;
    if (t == 0) t = 1.0f;
    return mu - beta * ::logf(t);
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateLogisticEx(
  curandGeneratorEx_t generator, float* outputPtr, size_t num, float mu, float beta)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  logistic_t<float> func;
  func.mu   = mu;
  func.beta = beta;
  return curandimpl::dispatch_sample<logistic_t<float>, float>(gen, func, num, outputPtr);
}

template <>
struct logistic_t<double> {
  double mu, beta;

  template <typename gen_t>
  __forceinline__ __host__ __device__ double operator()(gen_t& gen)
  {
    float y = curand_uniform_double(&gen);  // y cannot be 0
    float t = 1.0 / y - 1.0;
    if (t == 0) t = 1.0;
    return mu - beta * ::log(t);
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateLogisticDoubleEx(
  curandGeneratorEx_t generator, double* outputPtr, size_t num, double mu, double beta)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  logistic_t<double> func;
  func.mu   = mu;
  func.beta = beta;
  return curandimpl::dispatch_sample<logistic_t<double>, double>(gen, func, num, outputPtr);
}