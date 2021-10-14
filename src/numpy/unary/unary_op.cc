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

#include "numpy/unary/unary_op.h"
#include "numpy/unary/unary_op_template.inl"

namespace legate {
namespace numpy {

using namespace Legion;

template <UnaryOpCode OP_CODE, LegateTypeCode CODE, int DIM>
struct UnaryOpImplBody<VariantKind::CPU, OP_CODE, CODE, DIM> {
  using OP  = UnaryOp<OP_CODE, CODE>;
  using ARG = typename OP::T;
  using RES = std::result_of_t<OP(ARG)>;

  void operator()(OP func,
                  AccessorWO<RES, DIM> out,
                  AccessorRO<ARG, DIM> in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    if (dense) {
      auto outptr = out.ptr(rect);
      auto inptr  = in.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) outptr[idx] = func(inptr[idx]);
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        out[p] = func(in[p]);
      }
    }
  }
};

/*static*/ void UnaryOpTask::cpu_variant(TaskContext& context)
{
  unary_op_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { UnaryOpTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
