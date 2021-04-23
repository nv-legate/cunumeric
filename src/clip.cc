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

#include "clip.h"

namespace legate {
namespace numpy {
// Instantiate Clip's tasks for all Legate types
template class ClipTask<__half>;
template class ClipTask<float>;
template class ClipTask<double>;
template class ClipTask<int16_t>;
template class ClipTask<int32_t>;
template class ClipTask<int64_t>;
template class ClipTask<uint16_t>;
template class ClipTask<uint32_t>;
template class ClipTask<uint64_t>;
template class ClipTask<bool>;
template class ClipTask<complex<float>>;
template class ClipTask<complex<double>>;

template class ClipInplace<__half>;
template class ClipInplace<float>;
template class ClipInplace<double>;
template class ClipInplace<int16_t>;
template class ClipInplace<int32_t>;
template class ClipInplace<int64_t>;
template class ClipInplace<uint16_t>;
template class ClipInplace<uint32_t>;
template class ClipInplace<uint64_t>;
template class ClipInplace<bool>;
template class ClipInplace<complex<float>>;
template class ClipInplace<complex<double>>;

template class ClipScalar<__half>;
template class ClipScalar<float>;
template class ClipScalar<double>;
template class ClipScalar<int16_t>;
template class ClipScalar<int32_t>;
template class ClipScalar<int64_t>;
template class ClipScalar<uint16_t>;
template class ClipScalar<uint32_t>;
template class ClipScalar<uint64_t>;
template class ClipScalar<bool>;
template class ClipScalar<complex<float>>;
template class ClipScalar<complex<double>>;
}  // namespace numpy
}  // namespace legate
