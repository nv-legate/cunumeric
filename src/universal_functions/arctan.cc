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

#include "arctan.h"

namespace legate {
namespace numpy {
// instantiate ArcTan's tasks for all Legate types
template void ArcTan<__half>::instantiate_tasks();
template void ArcTan<float>::instantiate_tasks();
template void ArcTan<double>::instantiate_tasks();
template void ArcTan<int16_t>::instantiate_tasks();
template void ArcTan<int32_t>::instantiate_tasks();
template void ArcTan<int64_t>::instantiate_tasks();
template void ArcTan<uint16_t>::instantiate_tasks();
template void ArcTan<uint32_t>::instantiate_tasks();
template void ArcTan<uint64_t>::instantiate_tasks();
template void ArcTan<bool>::instantiate_tasks();
template void ArcTan<complex<float>>::instantiate_tasks();
template void ArcTan<complex<double>>::instantiate_tasks();
}  // namespace numpy
}  // namespace legate
