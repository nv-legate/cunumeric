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

#include "copy.h"

// instantiate CopyTask for all Legate types
namespace legate {
namespace numpy {
template class CopyTask<__half>;
template class CopyTask<float>;
template class CopyTask<double>;
template class CopyTask<int16_t>;
template class CopyTask<int32_t>;
template class CopyTask<int64_t>;
template class CopyTask<uint16_t>;
template class CopyTask<uint32_t>;
template class CopyTask<uint64_t>;
template class CopyTask<bool>;
template class CopyTask<complex<float>>;
template class CopyTask<complex<double>>;
}  // namespace numpy
}  // namespace legate
