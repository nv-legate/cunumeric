/* Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef __NUMPY_CONVERT_H__
#define __NUMPY_CONVERT_H__

#include "unary_operation.h"

namespace legate {
namespace numpy {

template<class To, class From>
__CUDA_HD__ To convert(const From& arg) {
  return static_cast<To>(arg);
}

// Specializations for complex numbers
#define COMPLEX_SPECIALIZATION(T)                                                    \
  template<>                                                                         \
  __CUDA_HD__ inline complex<T> convert<complex<T>, __half>(const __half& arg) {     \
    return complex<T>(static_cast<T>(arg));                                          \
  }                                                                                  \
  template<>                                                                         \
  __CUDA_HD__ inline complex<T> convert<complex<T>, float>(const float& arg) {       \
    return complex<T>(static_cast<T>(arg));                                          \
  }                                                                                  \
  template<>                                                                         \
  __CUDA_HD__ inline complex<T> convert<complex<T>, double>(const double& arg) {     \
    return complex<T>(static_cast<T>(arg));                                          \
  }                                                                                  \
  template<>                                                                         \
  __CUDA_HD__ inline complex<T> convert<complex<T>, uint16_t>(const uint16_t& arg) { \
    return complex<T>(static_cast<T>(arg));                                          \
  }                                                                                  \
  template<>                                                                         \
  __CUDA_HD__ inline complex<T> convert<complex<T>, uint32_t>(const uint32_t& arg) { \
    return complex<T>(static_cast<T>(arg));                                          \
  }                                                                                  \
  template<>                                                                         \
  __CUDA_HD__ inline complex<T> convert<complex<T>, uint64_t>(const uint64_t& arg) { \
    return complex<T>(static_cast<T>(arg));                                          \
  }                                                                                  \
  template<>                                                                         \
  __CUDA_HD__ inline complex<T> convert<complex<T>, int16_t>(const int16_t& arg) {   \
    return complex<T>(static_cast<T>(arg));                                          \
  }                                                                                  \
  template<>                                                                         \
  __CUDA_HD__ inline complex<T> convert<complex<T>, int32_t>(const int32_t& arg) {   \
    return complex<T>(static_cast<T>(arg));                                          \
  }                                                                                  \
  template<>                                                                         \
  __CUDA_HD__ inline complex<T> convert<complex<T>, int64_t>(const int64_t& arg) {   \
    return complex<T>(static_cast<T>(arg));                                          \
  }                                                                                  \
  template<>                                                                         \
  __CUDA_HD__ inline complex<T> convert<complex<T>, bool>(const bool& arg) {         \
    return complex<T>(static_cast<T>(arg));                                          \
  }
COMPLEX_SPECIALIZATION(float)
COMPLEX_SPECIALIZATION(double)
#undef COMPLEX_SPECIALIZATION

template<>
__CUDA_HD__ inline complex<__half> convert<complex<__half>, complex<float>>(const complex<float>& arg) {
  return complex<__half>(static_cast<__half>(arg.real()), static_cast<__half>(arg.imag()));
}

template<>
__CUDA_HD__ inline complex<__half> convert<complex<__half>, complex<double>>(const complex<double>& arg) {
  return complex<__half>(static_cast<__half>(arg.real()), static_cast<__half>(arg.imag()));
}

template<>
__CUDA_HD__ inline complex<float> convert<complex<float>, complex<__half>>(const complex<__half>& arg) {
  return complex<float>(arg.real(), arg.imag());
}

template<>
__CUDA_HD__ inline complex<float> convert<complex<float>, complex<double>>(const complex<double>& arg) {
  return complex<float>(arg.real(), arg.imag());
}

template<>
__CUDA_HD__ inline complex<double> convert<complex<double>, complex<__half>>(const complex<__half>& arg) {
  return complex<double>(arg.real(), arg.imag());
}

template<>
__CUDA_HD__ inline complex<double> convert<complex<double>, complex<float>>(const complex<float>& arg) {
  return complex<double>(arg.real(), arg.imag());
}

template<>
__CUDA_HD__ inline int64_t convert<int64_t, __half>(const __half& arg) {
#ifdef __CUDACC__
  // XXX cast to workaround nvbug 2613890
  return static_cast<float>(arg);
#else
  return arg;
#endif
}

template<>
__CUDA_HD__ inline uint64_t convert<uint64_t, __half>(const __half& arg) {
#ifdef __CUDACC__
  // XXX cast to workaround nvbug 2613890
  return static_cast<float>(arg);
#else
  return arg;
#endif
}

template<>
__CUDA_HD__ inline __half convert<__half, int64_t>(const int64_t& arg) {
#ifdef __CUDACC__
  // XXX cast to workaround nvbug 2613890
  return static_cast<float>(arg);
#else
  return static_cast<__half>(arg);
#endif
}

template<>
__CUDA_HD__ inline __half convert<__half, uint64_t>(const uint64_t& arg) {
#ifdef __CUDACC__
  // XXX cast to workaround nvbug 2613890
  return static_cast<float>(arg);
#else
  return static_cast<__half>(arg);
#endif
}

template<class To, class From>
struct ConvertOperation {
  constexpr static auto op_code = NumPyOpCode::NUMPY_CONVERT;
  using argument_type           = From;

  __CUDA_HD__
  To operator()(const argument_type& arg) const {
    // XXX this should just return arg when nvbug 2613890 has been fixed
    return convert<To>(arg);
  }
};

template<class To, class From>
class ConvertTask : public UnaryOperationTask<ConvertTask<To, From>, ConvertOperation<To, From>> {};

// specialize task_id for CONVERT
template<NumPyVariantCode variant_code, class From, class To>
constexpr int                                       task_id<NumPyOpCode::NUMPY_CONVERT, variant_code, To, From> =
    NUMPY_CONVERT_OFFSET + legate_type_code_of<To>* NUMPY_TYPE_OFFSET + variant_code +
    legate_type_code_of<From>*                      NUMPY_MAX_VARIANTS;

}    // namespace numpy
}    // namespace legate

#endif    // __NUMPY_CONVERT_H__
