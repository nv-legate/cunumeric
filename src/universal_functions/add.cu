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

#include "add.h"

namespace legate {
namespace numpy {
// Instantiate Add's tasks' gpu variants
template void Add<__half>::instantiate_task_gpu_variants();
template void Add<float>::instantiate_task_gpu_variants();
template void Add<double>::instantiate_task_gpu_variants();
template void Add<int16_t>::instantiate_task_gpu_variants();
template void Add<int32_t>::instantiate_task_gpu_variants();
template void Add<int64_t>::instantiate_task_gpu_variants();
template void Add<uint16_t>::instantiate_task_gpu_variants();
template void Add<uint32_t>::instantiate_task_gpu_variants();
template void Add<uint64_t>::instantiate_task_gpu_variants();
template void Add<bool>::instantiate_task_gpu_variants();
template void Add<complex<float>>::instantiate_task_gpu_variants();
template void Add<complex<double>>::instantiate_task_gpu_variants();
}    // namespace numpy
}    // namespace legate
