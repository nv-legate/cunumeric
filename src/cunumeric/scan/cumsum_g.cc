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

#include "cunumeric/scan/cumsum_g.h"
#include "cunumeric/scan/cumsum_g_template.inl"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>


namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE, int DIM>
struct Cumsum_gImplBody<VariantKind::CPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;
  
  struct add_scalar_funct
  {
    VAL V;
    add_scalar_funct(VAL a) : V(a);
    
    __host__ __device__
    void operator()(VAL &x)
    {
      x += V;
    }
  };
  
  size_t operator()(const AccessorWO<VAL, DIM>& out,
		    const AccessorRO<VAL, DIM>& in,
		    const AccessorRO<VAL, DIM>& sum_vals,
                    const Pitches<DIM - 1>& pitches,
                    const Rect<DIM>& rect,
		    const int axis,
		    const DomainPoint& partition_index)
  {
    auto outptr = out.ptr(rect.lo);
    auto inptr = in.ptr(rect.lo);
    auto sum_valsptr = sum_vals.ptr(???); // RRRR it's a broadcast, how do access?
    auto volume = rect.volume();
    if(axis == -1){
      // flattened scan (1D or no axis)
      if (patrition_index == 0){ // RRRR in condition correct?
	// first partition has nothing to do and can return;
	return;
      }
      // calculate base (sum up to partition_index-1)
      auto base = thrust::reduce(thrust::host, sum_valsptr, sum_valsptr + paratition_index - 1); // RRRR is the indexing format correct?

      // add base to out
      thrust::for_each(thrust::host, outptr, outptr + volume, add_scalar_funct(base));
    } else {
      // ND scan
      if (patrition_index[DIM - 1] == 0){ // RRRR in condition correct?
	// first patition has nothing to do and can return;
	return;
      }
      auto stride = rect.hi[DIM - 1] - rect.lo[DIM - 1] + 1;
      for(unit3264_t index = 0; index < volume; index += stride){
	// calculate base (sum up to partition_index-1)

	auto  base = thrust::reduce(thrust::host, sum_valsptr[???], sum_valsptr[???] + partition_index - 1); // RRRR is the indexing format correct?

	// add base to out
	thrust::for_each(thrust::host, outptr + index, outptr + index + stride, add_scalar_funct(base));
      }
    }
  }
};

/*static*/ void Cumsum_gTask::cpu_variant(TaskContext& context)
{
  cumsum_g_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { NonzeroTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
  
