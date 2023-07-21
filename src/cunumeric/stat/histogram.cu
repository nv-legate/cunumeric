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

#include "cunumeric/stat/histogram.h"
#include "cunumeric/stat/histogram_template.inl"

#include "cunumeric/cuda_help.h"

#include "cunumeric/stat/histogram.cuh"
#include "cunumeric/stat/histogram_impl.h"

#include "cunumeric/utilities/thrust_util.h"

#include <tuple>

// #define _DEBUG
#ifdef _DEBUG
#include <iostream>
#include <iterator>
#include <algorithm>
#include <numeric>

#include <thrust/host_vector.h>
#endif

namespace cunumeric {

template <Type::Code CODE>
struct HistogramImplBody<VariantKind::GPU, CODE> {
  using VAL = legate_type_of<CODE>;

  // for now, it has been decided to hardcode these types:
  //
  using BinType    = double;
  using WeightType = double;

  // in the future we might relax relax that requirement,
  // but complicate dispatching:
  //
  // template <typename BinType = VAL, typename WeightType = VAL>
  void operator()(const AccessorRO<VAL, 1>& src,
                  const Rect<1>& src_rect,
                  const AccessorRO<BinType, 1>& bins,
                  const Rect<1>& bins_rect,
                  const AccessorRO<WeightType, 1>& weights,
                  const Rect<1>& weights_rect,
                  const AccessorRD<SumReduction<WeightType>, true, 1>& result,
                  const Rect<1>& result_rect) const
  {
    auto stream          = get_cached_stream();
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    auto exe_pol         = DEFAULT_POLICY.on(stream);

    detail::histogram_wrapper(
      exe_pol, src, src_rect, bins, bins_rect, weights, weights_rect, result, result_rect, stream_);
  }
};

/*static*/ void HistogramTask::gpu_variant(TaskContext& context)
{
  histogram_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
