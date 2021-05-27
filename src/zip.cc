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

#include "zip.h"

namespace legate {
namespace numpy {
// Instantiate tasks for zipping multiple arrays of coordinates into one array of points,
// up to our maximum array dimensionality.
#define DIMFUNC(N) template class ZipTask<N>;
LEGATE_FOREACH_N(DIMFUNC)
#undef DIMFUNC
}  // namespace numpy
}  // namespace legate
