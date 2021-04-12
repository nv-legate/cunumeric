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

#include "inplace_div.h"

// instantiate InplaceDivTask for the types Legate handles
template class InplaceDivTask<float>;
template class InplaceDivTask<double>;
template class InplaceDivTask<int16_t>;
template class InplaceDivTask<int32_t>;
template class InplaceDivTask<int64_t>;
template class InplaceDivTask<uint16_t>;
template class InplaceDivTask<uint32_t>;
template class InplaceDivTask<uint64_t>;
template class InplaceDivTask<bool>;
template class InplaceDivTask<__half>;
