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

#include "inplace_div_broadcast.h"

// XXX can we find a way to avoid this?
template void PointTask<InplaceDivBroadcastTask<float>>::gpu_variant(
  const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<InplaceDivBroadcastTask<double>>::gpu_variant(
  const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<InplaceDivBroadcastTask<int16_t>>::gpu_variant(
  const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<InplaceDivBroadcastTask<int32_t>>::gpu_variant(
  const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<InplaceDivBroadcastTask<int64_t>>::gpu_variant(
  const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<InplaceDivBroadcastTask<uint16_t>>::gpu_variant(
  const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<InplaceDivBroadcastTask<uint32_t>>::gpu_variant(
  const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<InplaceDivBroadcastTask<uint64_t>>::gpu_variant(
  const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<InplaceDivBroadcastTask<bool>>::gpu_variant(
  const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<InplaceDivBroadcastTask<__half>>::gpu_variant(
  const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<InplaceDivBroadcastTask<complex<float>>>::gpu_variant(
  const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<InplaceDivBroadcastTask<complex<double>>>::gpu_variant(
  const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
