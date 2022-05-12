/* Copyright 2022 NVIDIA Corporation
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

#include "cunumeric/sort/thrust_sort.cuh"

namespace cunumeric {

void thrust_local_sort(const complex<double>* values_in,
                       complex<double>* values_out,
                       const int64_t* indices_in,
                       int64_t* indices_out,
                       const size_t volume,
                       const size_t sort_dim_size,
                       const bool stable,
                       cudaStream_t stream)
{
  detail::thrust_local_sort(
    values_in, values_out, indices_in, indices_out, volume, sort_dim_size, stable, stream);
}

}  // namespace cunumeric
