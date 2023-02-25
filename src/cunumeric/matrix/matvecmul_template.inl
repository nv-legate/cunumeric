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

#pragma once

// Useful for IDEs
#include "cunumeric/matrix/matvecmul.h"
#include "cunumeric/matrix/util.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, LegateTypeCode CODE>
struct MatVecMulImplBody;

template <LegateTypeCode CODE>
struct support_matvecmul : std::false_type {};
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
template <>
struct support_matvecmul<LegateTypeCode::COMPLEX64_LT> : std::true_type {
  using ACC_TYPE = complex<float>;
};
template <>
struct support_matvecmul<LegateTypeCode::COMPLEX128_LT> : std::true_type {
  using ACC_TYPE = complex<double>;
};

template <VariantKind KIND>
struct MatVecMulImpl {
  template <LegateTypeCode CODE, std::enable_if_t<support_matvecmul<CODE>::value>* = nullptr>
  void operator()(MatVecMulArgs& args) const
  {
    using VAL = legate_type_of<CODE>;
    using ACC = typename support_matvecmul<CODE>::ACC_TYPE;

    auto shape = args.rhs1.shape<2>().intersection(args.rhs2.shape<2>());

    if (shape.empty()) return;

    auto m = static_cast<size_t>(shape.hi[0] - shape.lo[0] + 1);
    auto n = static_cast<size_t>(shape.hi[1] - shape.lo[1] + 1);

    size_t mat_strides[2];
    size_t vec_strides[2];
    const VAL* mat = args.rhs1.read_accessor<VAL, 2>(shape).ptr(shape, mat_strides);
    const VAL* vec = args.rhs2.read_accessor<VAL, 2>(shape).ptr(shape, vec_strides);

    bool transpose_mat;
    size_t mat_stride = stride_for_blas(m, n, mat_strides[0], mat_strides[1], transpose_mat);
    if (transpose_mat) std::swap(m, n);

    size_t lhs_strides[2];
    auto lhs = args.lhs.reduce_accessor<SumReduction<ACC>, true, 2>().ptr(shape, lhs_strides);

#ifdef DEBUG_CUNUMERIC
    assert(vec_strides[0] == 0 && vec_strides[1] == 1);
    assert(lhs_strides[0] == 1 && lhs_strides[1] == 0);
#endif

    MatVecMulImplBody<KIND, CODE>()(
      m, n, lhs, mat, vec, mat_stride, transpose_mat, args.lhs.is_readable());
  }

  template <LegateTypeCode CODE, std::enable_if_t<!support_matvecmul<CODE>::value>* = nullptr>
  void operator()(MatVecMulArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void matvecmul_template(TaskContext& context)
{
  auto& reductions = context.reductions();
  auto& inputs     = context.inputs();

  MatVecMulArgs args{reductions[0], inputs[0], inputs[1]};
  // Note that we can't dispatch on the lhs's type,
  // as the lhs can have a different type than the rhs'
  type_dispatch(args.rhs1.code(), MatVecMulImpl<KIND>{}, args);
}

}  // namespace cunumeric
