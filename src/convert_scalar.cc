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

#include "convert_scalar.h"

// instantiate ConvertScalarTask for the cross product of types Legate handles
// we omit the T1 == T2 case

namespace legate {
namespace numpy {
// T1 == __half
template class ConvertScalarTask<__half, float>;
template class ConvertScalarTask<__half, double>;
template class ConvertScalarTask<__half, int16_t>;
template class ConvertScalarTask<__half, int32_t>;
template class ConvertScalarTask<__half, int64_t>;
template class ConvertScalarTask<__half, uint16_t>;
template class ConvertScalarTask<__half, uint32_t>;
template class ConvertScalarTask<__half, uint64_t>;
template class ConvertScalarTask<__half, bool>;

// T1 == float
template class ConvertScalarTask<float, __half>;
template class ConvertScalarTask<float, double>;
template class ConvertScalarTask<float, int16_t>;
template class ConvertScalarTask<float, int32_t>;
template class ConvertScalarTask<float, int64_t>;
template class ConvertScalarTask<float, uint16_t>;
template class ConvertScalarTask<float, uint32_t>;
template class ConvertScalarTask<float, uint64_t>;
template class ConvertScalarTask<float, bool>;

// T1 == double
template class ConvertScalarTask<double, __half>;
template class ConvertScalarTask<double, float>;
template class ConvertScalarTask<double, int16_t>;
template class ConvertScalarTask<double, int32_t>;
template class ConvertScalarTask<double, int64_t>;
template class ConvertScalarTask<double, uint16_t>;
template class ConvertScalarTask<double, uint32_t>;
template class ConvertScalarTask<double, uint64_t>;
template class ConvertScalarTask<double, bool>;

// T1 == int16_t
template class ConvertScalarTask<int16_t, __half>;
template class ConvertScalarTask<int16_t, float>;
template class ConvertScalarTask<int16_t, double>;
template class ConvertScalarTask<int16_t, int32_t>;
template class ConvertScalarTask<int16_t, int64_t>;
template class ConvertScalarTask<int16_t, uint16_t>;
template class ConvertScalarTask<int16_t, uint32_t>;
template class ConvertScalarTask<int16_t, uint64_t>;
template class ConvertScalarTask<int16_t, bool>;

// T1 == int32_t
template class ConvertScalarTask<int32_t, __half>;
template class ConvertScalarTask<int32_t, float>;
template class ConvertScalarTask<int32_t, double>;
template class ConvertScalarTask<int32_t, int16_t>;
template class ConvertScalarTask<int32_t, int64_t>;
template class ConvertScalarTask<int32_t, uint16_t>;
template class ConvertScalarTask<int32_t, uint32_t>;
template class ConvertScalarTask<int32_t, uint64_t>;
template class ConvertScalarTask<int32_t, bool>;

// T1 == int64_t
template class ConvertScalarTask<int64_t, __half>;
template class ConvertScalarTask<int64_t, float>;
template class ConvertScalarTask<int64_t, double>;
template class ConvertScalarTask<int64_t, int16_t>;
template class ConvertScalarTask<int64_t, int32_t>;
template class ConvertScalarTask<int64_t, uint16_t>;
template class ConvertScalarTask<int64_t, uint32_t>;
template class ConvertScalarTask<int64_t, uint64_t>;
template class ConvertScalarTask<int64_t, bool>;

// T1 == uint16_t
template class ConvertScalarTask<uint16_t, __half>;
template class ConvertScalarTask<uint16_t, float>;
template class ConvertScalarTask<uint16_t, double>;
template class ConvertScalarTask<uint16_t, int16_t>;
template class ConvertScalarTask<uint16_t, int32_t>;
template class ConvertScalarTask<uint16_t, int64_t>;
template class ConvertScalarTask<uint16_t, uint32_t>;
template class ConvertScalarTask<uint16_t, uint64_t>;
template class ConvertScalarTask<uint16_t, bool>;

// T1 == uint32_t
template class ConvertScalarTask<uint32_t, __half>;
template class ConvertScalarTask<uint32_t, float>;
template class ConvertScalarTask<uint32_t, double>;
template class ConvertScalarTask<uint32_t, int16_t>;
template class ConvertScalarTask<uint32_t, int32_t>;
template class ConvertScalarTask<uint32_t, int64_t>;
template class ConvertScalarTask<uint32_t, uint16_t>;
template class ConvertScalarTask<uint32_t, uint64_t>;
template class ConvertScalarTask<uint32_t, bool>;

// T1 == uint64_t
template class ConvertScalarTask<uint64_t, __half>;
template class ConvertScalarTask<uint64_t, float>;
template class ConvertScalarTask<uint64_t, double>;
template class ConvertScalarTask<uint64_t, int16_t>;
template class ConvertScalarTask<uint64_t, int32_t>;
template class ConvertScalarTask<uint64_t, int64_t>;
template class ConvertScalarTask<uint64_t, uint16_t>;
template class ConvertScalarTask<uint64_t, uint32_t>;
template class ConvertScalarTask<uint64_t, bool>;

// T1 == bool
template class ConvertScalarTask<bool, __half>;
template class ConvertScalarTask<bool, float>;
template class ConvertScalarTask<bool, double>;
template class ConvertScalarTask<bool, int16_t>;
template class ConvertScalarTask<bool, int32_t>;
template class ConvertScalarTask<bool, int64_t>;
template class ConvertScalarTask<bool, uint16_t>;
template class ConvertScalarTask<bool, uint32_t>;
template class ConvertScalarTask<bool, uint64_t>;

// T1 == complex64
template class ConvertScalarTask<complex<float>, __half>;
template class ConvertScalarTask<complex<float>, float>;
template class ConvertScalarTask<complex<float>, double>;
template class ConvertScalarTask<complex<float>, int16_t>;
template class ConvertScalarTask<complex<float>, int32_t>;
template class ConvertScalarTask<complex<float>, int64_t>;
template class ConvertScalarTask<complex<float>, uint16_t>;
template class ConvertScalarTask<complex<float>, uint32_t>;
template class ConvertScalarTask<complex<float>, uint64_t>;
template class ConvertScalarTask<complex<float>, bool>;
template class ConvertScalarTask<complex<float>, complex<double>>;

// T1 == complex128
template class ConvertScalarTask<complex<double>, __half>;
template class ConvertScalarTask<complex<double>, float>;
template class ConvertScalarTask<complex<double>, double>;
template class ConvertScalarTask<complex<double>, int16_t>;
template class ConvertScalarTask<complex<double>, int32_t>;
template class ConvertScalarTask<complex<double>, int64_t>;
template class ConvertScalarTask<complex<double>, uint16_t>;
template class ConvertScalarTask<complex<double>, uint32_t>;
template class ConvertScalarTask<complex<double>, uint64_t>;
template class ConvertScalarTask<complex<double>, bool>;
template class ConvertScalarTask<complex<double>, complex<float>>;

}  // namespace numpy
}  // namespace legate
