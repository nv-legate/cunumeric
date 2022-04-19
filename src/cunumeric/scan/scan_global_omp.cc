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

#include "cunumeric/scan/scan_global.h"
#include "cunumeric/scan/scan_global_template.inl"

#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>
#include <omp.h>


namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE, int DIM>
struct ScanGlobalImplBody<VariantKind::OMP, CODE, DIM> {
  using VAL = legate_type_of<CODE>;
  
  struct add_scalar_funct
  {
    VAL V;
    add_scalar_funct(VAL a) : V(a) {}
    
    __host__ __device__
    void operator()(VAL &x)
    {
      x = x + V;
    }
  };
  
  struct prod_scalar_funct
  {
    VAL V;
    prod_scalar_funct(VAL a) : V(a) {}
    
    __host__ __device__
    void operator()(VAL &x)
    {
      x = x * V;
    }
  };
  
  void operator()(const AccessorRW<VAL, DIM>& out,
		  const AccessorRO<VAL, DIM>& sum_vals,
		  const Pitches<DIM - 1>& out_pitches,
		  const Rect<DIM>& out_rect,
		  const Pitches<DIM - 1>& sum_vals_pitches,
		  const Rect<DIM>& sum_vals_rect,
		  const DomainPoint& partition_index,
		  const int prod)
  {
    auto outptr = out.ptr(out_rect.lo);
    auto volume = out_rect.volume();

    if (partition_index[DIM - 1] == 0){
      // first partition has nothing to do and can return;
      return;
    }

    auto stride = out_rect.hi[DIM - 1] - out_rect.lo[DIM - 1] + 1;
    for(uint64_t index = 0; index < volume; index += stride){
      // get the corresponding ND index with base zero to use for sum_val
      auto sum_valsp = out_pitches.unflatten(index, out_rect.lo) - out_rect.lo;
      // first element on scan axis
      sum_valsp[DIM - 1] = 0;
      if(prod == 0){
	// calculate scan up to partition_index-1
	auto base = thrust::reduce(thrust::omp::par, &sum_vals[sum_valsp], &sum_vals[sum_valsp] + partition_index[DIM - 1] - 1); // RRRR is the indexing format correct?

	// apply base to out
	thrust::for_each(thrust::omp::par, outptr + index, outptr + index + stride, add_scalar_funct(base));
      } else {
	auto base = thrust::reduce(thrust::omp::par, &sum_vals[sum_valsp], &sum_vals[sum_valsp] + partition_index[DIM - 1] - 1, (VAL)1, thrust::multiplies<VAL>()); // RRRR is the indexing format correct?

	// apply base to out
	thrust::for_each(thrust::omp::par, outptr + index, outptr + index + stride, prod_scalar_funct(base));
      }
    }
  }
};

/*static*/ void ScanGlobalTask::omp_variant(TaskContext& context)
{
  scan_global_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
  
