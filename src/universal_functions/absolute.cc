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

#include "absolute.h"

namespace legate {
namespace numpy {
// instantiate Absolute's tasks for all Legate types
template void Absolute<__half>::instantiate_tasks();
template void Absolute<float>::instantiate_tasks();
template void Absolute<double>::instantiate_tasks();
template void Absolute<int16_t>::instantiate_tasks();
template void Absolute<int32_t>::instantiate_tasks();
template void Absolute<int64_t>::instantiate_tasks();
template void Absolute<uint16_t>::instantiate_tasks();
template void Absolute<uint32_t>::instantiate_tasks();
template void Absolute<uint64_t>::instantiate_tasks();
template void Absolute<bool>::instantiate_tasks();
template void Absolute<complex<float>>::instantiate_tasks();
template void Absolute<complex<double>>::instantiate_tasks();
}    // namespace numpy
}    // namespace legate
