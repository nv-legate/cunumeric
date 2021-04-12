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

using namespace Legion;

namespace legate {
namespace numpy {

template void PointTask<FillTask<__half>>::gpu_variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<FillTask<float>>::gpu_variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<FillTask<double>>::gpu_variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<FillTask<int16_t>>::gpu_variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<FillTask<int32_t>>::gpu_variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<FillTask<int64_t>>::gpu_variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<FillTask<uint16_t>>::gpu_variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<FillTask<uint32_t>>::gpu_variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<FillTask<uint64_t>>::gpu_variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<FillTask<bool>>::gpu_variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<FillTask<complex<float>>>::gpu_variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<FillTask<complex<double>>>::gpu_variant(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
}    // namespace numpy
}    // namespace legate
