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

#include "core.h"
#include "deserializer.h"
#include "dispatch.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <VariantKind KIND, LegateTypeCode CODE>
struct MatVecMulImplBody;

template <LegateTypeCode CODE>
struct support_matvecmul : std::false_type {
};
template <>
struct support_matvecmul<LegateTypeCode::DOUBLE_LT> : std::true_type {
  using ACC_TYPE = double;
};
template <>
struct support_matvecmul<LegateTypeCode::FLOAT_LT> : std::true_type {
  using ACC_TYPE = float;
};
template <>
struct support_matvecmul<LegateTypeCode::HALF_LT> : std::true_type {
  using ACC_TYPE = float;
};

template <VariantKind KIND>
struct MatVecMulImpl {
  template <LegateTypeCode CODE, std::enable_if_t<support_matvecmul<CODE>::value> * = nullptr>
  void operator()(MatVecMulArgs &args) const
  {
    using VAL = legate_type_of<CODE>;
    using ACC = typename support_matvecmul<CODE>::ACC_TYPE;

    auto shape = args.rhs1.shape<2>().intersection(args.rhs2.shape<2>());
    auto m     = static_cast<size_t>(shape.hi[0] - shape.lo[0] + 1);
    auto n     = static_cast<size_t>(shape.hi[1] - shape.lo[1] + 1);

    size_t mat_stride  = 0;
    bool transpose_mat = false;
    const VAL *mat     = nullptr;
    const VAL *vec     = nullptr;

    size_t mat_strides[2];
    size_t vec_strides[2];
    if (args.left_matrix) {
      // M * v
      mat           = args.rhs1.read_accessor<VAL, 2>(shape).ptr(shape, mat_strides);
      vec           = args.rhs2.read_accessor<VAL, 2>(shape).ptr(shape, vec_strides);
      mat_stride    = std::max(mat_strides[0], mat_strides[1]);
      transpose_mat = mat_strides[1] == mat_stride;
      if (transpose_mat) std::swap(m, n);
    } else {
      // (M^T * v)^T
      vec           = args.rhs1.read_accessor<VAL, 2>(shape).ptr(shape, vec_strides);
      mat           = args.rhs2.read_accessor<VAL, 2>(shape).ptr(shape, mat_strides);
      mat_stride    = std::max(mat_strides[0], mat_strides[1]);
      transpose_mat = mat_strides[0] == mat_stride;
      if (!transpose_mat) std::swap(m, n);
    }

    size_t lhs_strides[2];
    auto lhs = args.lhs.reduce_accessor<SumReduction<ACC>, true, 2>().ptr(shape, lhs_strides);

    MatVecMulImplBody<KIND, CODE>()(m, n, lhs, mat, vec, mat_stride, transpose_mat);
  }

  template <LegateTypeCode CODE, std::enable_if_t<!support_matvecmul<CODE>::value> * = nullptr>
  void operator()(MatVecMulArgs &args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void matvecmul_template(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context context,
                               Runtime *runtime)
{
  Deserializer ctx(task, regions);
  MatVecMulArgs args;
  deserialize(ctx, args);
  // Note that we can't dispatch on the lhs's type,
  // as the lhs can have a different type than the rhs'
  type_dispatch(args.rhs1.code(), MatVecMulImpl<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
