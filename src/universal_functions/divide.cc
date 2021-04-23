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

#include "divide.h"

namespace legate {
namespace numpy {
// instantiate Divide's tasks for all Legate types
template void Divide<__half>::instantiate_tasks();
template void Divide<float>::instantiate_tasks();
template void Divide<double>::instantiate_tasks();
template void Divide<int16_t>::instantiate_tasks();
template void Divide<int32_t>::instantiate_tasks();
template void Divide<int64_t>::instantiate_tasks();
template void Divide<uint16_t>::instantiate_tasks();
template void Divide<uint32_t>::instantiate_tasks();
template void Divide<uint64_t>::instantiate_tasks();
template void Divide<bool>::instantiate_tasks();
template void Divide<complex<float>>::instantiate_tasks();
template void Divide<complex<double>>::instantiate_tasks();
}  // namespace numpy
}  // namespace legate
