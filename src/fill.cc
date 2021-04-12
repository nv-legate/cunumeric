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

#include "fill.h"

// instantiate FillTask for all Legate types
namespace legate {
namespace numpy {
template class FillTask<__half>;
template class FillTask<float>;
template class FillTask<double>;
template class FillTask<int16_t>;
template class FillTask<int32_t>;
template class FillTask<int64_t>;
template class FillTask<uint16_t>;
template class FillTask<uint32_t>;
template class FillTask<uint64_t>;
template class FillTask<bool>;
}    // namespace numpy
}    // namespace legate
