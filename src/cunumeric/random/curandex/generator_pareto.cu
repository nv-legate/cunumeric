// (c) 2022 NVIDIA all rights reserved
#include "generator.cuh"

template <typename field_t>
struct pareto_t;

template <>
struct pareto_t<float> {
  float xm, invalpha;

  template <typename gen_t>
  __forceinline__ __host__ __device__ float operator()(gen_t& gen)
  {
    float y = curand_uniform(&gen);                    // y cannot be 0
    return xm * ::expf(-::logf(y) * invalpha) - 1.0f;  // here, use -1.0f to align with numpy
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateParetoEx(
  curandGeneratorEx_t generator, float* outputPtr, size_t num, float xm, float alpha)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  pareto_t<float> func;
  func.xm       = xm;
  func.invalpha = 1.0f / alpha;
  return curandimpl::dispatch_sample<pareto_t<float>, float>(gen, func, num, outputPtr);
}

template <>
struct pareto_t<double> {
  double xm, invalpha;

  template <typename gen_t>
  __forceinline__ __host__ __device__ double operator()(gen_t& gen)
  {
    double y = curand_uniform_double(&gen);         // y cannot be 0
    return xm * ::exp(-::log(y) * invalpha) - 1.0;  // here, use -1.0 to align with numpy
  }
};

extern "C" curandStatus_t CURANDAPI curandGenerateParetoDoubleEx(
  curandGeneratorEx_t generator, double* outputPtr, size_t num, double xm, double alpha)
{
  curandimpl::basegenerator* gen = (curandimpl::basegenerator*)generator;
  pareto_t<double> func;
  func.xm       = xm;
  func.invalpha = 1.0 / alpha;
  return curandimpl::dispatch_sample<pareto_t<double>, double>(gen, func, num, outputPtr);
}
