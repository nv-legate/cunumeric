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

#include "numpy/binary/binary_red.h"
#include "numpy/binary/binary_red_template.inl"

namespace legate {
namespace numpy {

using namespace Legion;

template <BinaryOpCode OP_CODE, LegateTypeCode CODE, int DIM>
struct BinaryRedImplBody<VariantKind::OMP, OP_CODE, CODE, DIM> {
  using OP  = BinaryOp<OP_CODE, CODE>;
  using ARG = legate_type_of<CODE>;

  template <typename AccessorRD>
  void operator()(OP func,
                  AccessorRD out,
                  AccessorRO<ARG, DIM> in1,
                  AccessorRO<ARG, DIM> in2,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    size_t volume = rect.volume();
    bool result   = true;
    if (dense) {
      auto in1ptr = in1.ptr(rect);
      auto in2ptr = in2.ptr(rect);
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx)
        if (!func(in1ptr[idx], in2ptr[idx])) result = false;
    } else {
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) {
        auto point = pitches.unflatten(idx, rect.lo);
        if (!func(in1[point], in2[point])) result = false;
      }
    }

    out.reduce(0, result);
  }
};

/*static*/ void BinaryRedTask::omp_variant(TaskContext& context)
{
  binary_red_template<VariantKind::OMP>(context);
}

}  // namespace numpy
}  // namespace legate
