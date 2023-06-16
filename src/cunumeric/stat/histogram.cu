/* Copyright 2023 NVIDIA Corporation
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

#include "cunumeric/stat/histogram.cuh"
#include "cunumeric/stat/histogram_impl.h"

#include "cunumeric/stat/histogram_template.inl"

namespace cunumeric {
template <Type::Code CODE>
struct HistogramImplBody<VariantKind::GPU, CODE> {
  using VAL = legate_type_of<CODE>;

  template <typename BinType, typename WeightType = VAL>
  void operator()(const AccessorRW<VAL, 1>& src,
                  const AccessorRO<BinType /*?*/, 1>& bins,
                  const AccessorRO<WeightType /*?*/, 1>& weights,
                  const AccessorRD<SumReduction<WeightType>, false, 1>& result) const
  {
    // TODO:
    //
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void HistogramTask::gpu_variant(TaskContext& context)
{
  bincount_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
