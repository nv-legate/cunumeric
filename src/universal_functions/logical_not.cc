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

#include "logical_not.h"

// instantiate LogicalNot's tasks for all Legate types
namespace legate {
namespace numpy {
template void LogicalNot<__half>::instantiate_tasks();
template void LogicalNot<float>::instantiate_tasks();
template void LogicalNot<double>::instantiate_tasks();
template void LogicalNot<int16_t>::instantiate_tasks();
template void LogicalNot<int32_t>::instantiate_tasks();
template void LogicalNot<int64_t>::instantiate_tasks();
template void LogicalNot<uint16_t>::instantiate_tasks();
template void LogicalNot<uint32_t>::instantiate_tasks();
template void LogicalNot<uint64_t>::instantiate_tasks();
template void LogicalNot<bool>::instantiate_tasks();
template void LogicalNot<complex<float>>::instantiate_tasks();
template void LogicalNot<complex<double>>::instantiate_tasks();
}  // namespace numpy
}  // namespace legate
