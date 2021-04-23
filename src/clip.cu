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

#include "clip.h"

using namespace Legion;

namespace legate {
namespace numpy {
// Instantiate Clip's tasks' gpu variants
template void PointTask<ClipTask<__half>>::gpu_variant(const Task*,
                                                       const std::vector<PhysicalRegion>&,
                                                       Context,
                                                       Runtime*);
template void PointTask<ClipTask<float>>::gpu_variant(const Task*,
                                                      const std::vector<PhysicalRegion>&,
                                                      Context,
                                                      Runtime*);
template void PointTask<ClipTask<double>>::gpu_variant(const Task*,
                                                       const std::vector<PhysicalRegion>&,
                                                       Context,
                                                       Runtime*);
template void PointTask<ClipTask<int16_t>>::gpu_variant(const Task*,
                                                        const std::vector<PhysicalRegion>&,
                                                        Context,
                                                        Runtime*);
template void PointTask<ClipTask<int32_t>>::gpu_variant(const Task*,
                                                        const std::vector<PhysicalRegion>&,
                                                        Context,
                                                        Runtime*);
template void PointTask<ClipTask<int64_t>>::gpu_variant(const Task*,
                                                        const std::vector<PhysicalRegion>&,
                                                        Context,
                                                        Runtime*);
template void PointTask<ClipTask<uint16_t>>::gpu_variant(const Task*,
                                                         const std::vector<PhysicalRegion>&,
                                                         Context,
                                                         Runtime*);
template void PointTask<ClipTask<uint32_t>>::gpu_variant(const Task*,
                                                         const std::vector<PhysicalRegion>&,
                                                         Context,
                                                         Runtime*);
template void PointTask<ClipTask<uint64_t>>::gpu_variant(const Task*,
                                                         const std::vector<PhysicalRegion>&,
                                                         Context,
                                                         Runtime*);
template void PointTask<ClipTask<bool>>::gpu_variant(const Task*,
                                                     const std::vector<PhysicalRegion>&,
                                                     Context,
                                                     Runtime*);
template void PointTask<ClipTask<complex<float>>>::gpu_variant(const Task*,
                                                               const std::vector<PhysicalRegion>&,
                                                               Context,
                                                               Runtime*);
template void PointTask<ClipTask<complex<double>>>::gpu_variant(const Task*,
                                                                const std::vector<PhysicalRegion>&,
                                                                Context,
                                                                Runtime*);

template void PointTask<ClipInplace<__half>>::gpu_variant(const Task*,
                                                          const std::vector<PhysicalRegion>&,
                                                          Context,
                                                          Runtime*);
template void PointTask<ClipInplace<float>>::gpu_variant(const Task*,
                                                         const std::vector<PhysicalRegion>&,
                                                         Context,
                                                         Runtime*);
template void PointTask<ClipInplace<double>>::gpu_variant(const Task*,
                                                          const std::vector<PhysicalRegion>&,
                                                          Context,
                                                          Runtime*);
template void PointTask<ClipInplace<int16_t>>::gpu_variant(const Task*,
                                                           const std::vector<PhysicalRegion>&,
                                                           Context,
                                                           Runtime*);
template void PointTask<ClipInplace<int32_t>>::gpu_variant(const Task*,
                                                           const std::vector<PhysicalRegion>&,
                                                           Context,
                                                           Runtime*);
template void PointTask<ClipInplace<int64_t>>::gpu_variant(const Task*,
                                                           const std::vector<PhysicalRegion>&,
                                                           Context,
                                                           Runtime*);
template void PointTask<ClipInplace<uint16_t>>::gpu_variant(const Task*,
                                                            const std::vector<PhysicalRegion>&,
                                                            Context,
                                                            Runtime*);
template void PointTask<ClipInplace<uint32_t>>::gpu_variant(const Task*,
                                                            const std::vector<PhysicalRegion>&,
                                                            Context,
                                                            Runtime*);
template void PointTask<ClipInplace<uint64_t>>::gpu_variant(const Task*,
                                                            const std::vector<PhysicalRegion>&,
                                                            Context,
                                                            Runtime*);
template void PointTask<ClipInplace<bool>>::gpu_variant(const Task*,
                                                        const std::vector<PhysicalRegion>&,
                                                        Context,
                                                        Runtime*);
template void PointTask<ClipInplace<complex<float>>>::gpu_variant(
  const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
template void PointTask<ClipInplace<complex<double>>>::gpu_variant(
  const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);

}  // namespace numpy
}  // namespace legate
