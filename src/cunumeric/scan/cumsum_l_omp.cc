/* Copyright 2021-2022 NVIDIA Corporation
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

#include "cunumeric/scan/cumsum_l.h"
#include "cunumeric/scan/cumsum_l_template.inl"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>
#include <omp.h>


namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE, int DIM>
struct Cumsum_lImplBody<VariantKind::OMP, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  size_t operator()(const AccessorWO<VAL, DIM>& out,
		    const AccessorRO<VAL, DIM>& in,
		    Array& sum_vals,
                    const Pitches<DIM - 1>& pitches,
                    const Rect<DIM>& rect)
  {
    auto outptr = out.ptr(rect.lo);
    auto inptr = in.ptr(rect.lo);
    auto volume = rect.volume();
    
    auto stride = rect.hi[DIM - 1] - rect.lo[DIM - 1] + 1;
    auto iters = volume / iters;

    Point<DIM> extents = rect.hi - rect.lo + Point<DIM>::ONES();
    extents[DIM - 1] = 1; // one element along scan axis

    auto sum_valsptr = sum_vals.create_output_buffer<VAL, DIM>(extents, true);

    for(unit64_t index = 0; index < volume; index += stride){
      thrust::inclusive_scan(thrust::omp::par, inptr + index, inptr + index + stride, outptr + index);
      // get the corresponding ND index with base zero to use for sum_val
      auto sum_valp = pitches.unflatten(index, rect.lo) - rect.lo;
      // only one element on scan axis
      sum_valp[DIM - 1] = 0;
      // write out the partition sum
      sum_valsptr[sum_valp] = outptr[index + stride - 1];
    }
  }
};

/*static*/ void Cumsum_lTask::omp_variant(TaskContext& context)
{
  cumsum_l_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
  
