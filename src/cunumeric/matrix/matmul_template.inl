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
#include "cunumeric/matrix/matmul.h"
#include "cunumeric/matrix/util.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct MatMulImplBody;

template <Type::Code CODE>
struct support_matmul : std::false_type {};
template <>
struct support_matmul<Type::Code::FLOAT64> : std::true_type {
  using ACC_TYPE = double;
};
template <>
struct support_matmul<Type::Code::FLOAT32> : std::true_type {
  using ACC_TYPE = float;
};
template <>
struct support_matmul<Type::Code::FLOAT16> : std::true_type {
  using ACC_TYPE = float;
};
template <>
struct support_matmul<Type::Code::COMPLEX64> : std::true_type {
  using ACC_TYPE = complex<float>;
};
template <>
struct support_matmul<Type::Code::COMPLEX128> : std::true_type {
  using ACC_TYPE = complex<double>;
};

template <VariantKind KIND>
struct MatMulImpl {
  template <Type::Code CODE, std::enable_if_t<support_matmul<CODE>::value>* = nullptr>
  void operator()(MatMulArgs& args) const
  {
    using VAL = legate_type_of<CODE>;
    using ACC = typename support_matmul<CODE>::ACC_TYPE;

    // Note that rhs1 and rhs2 may have different shapes. Here's why: rhs1 and rhs2 are promoted
    // on one of their dimensions, and in case that the promoted dimension is partitioned,
    // the store cannot see that partitioning, because that dimension doesn't map to the store's
    // original domain whose partitioning is only what the store can observe. Therefore, we must
    // take an intersection of the rhs1's and rhs2's shapes to get a correct "active" area
    // in their bloated domains.
    auto shape = args.rhs1.shape<3>().intersection(args.rhs2.shape<3>());

    if (shape.empty()) return;

    const auto m = shape.hi[0] - shape.lo[0] + 1;
    const auto k = shape.hi[1] - shape.lo[1] + 1;
    const auto n = shape.hi[2] - shape.lo[2] + 1;

    size_t lhs_strides[3];
    size_t rhs1_strides[3];
    size_t rhs2_strides[3];

    auto rhs1 = args.rhs1.read_accessor<VAL, 3>(shape).ptr(shape, rhs1_strides);
    auto rhs2 = args.rhs2.read_accessor<VAL, 3>(shape).ptr(shape, rhs2_strides);
    auto lhs  = args.lhs.reduce_accessor<SumReduction<ACC>, true, 3>(shape).ptr(shape, lhs_strides);

#ifdef DEBUG_CUNUMERIC
    assert(rhs1_strides[2] == 0);
    assert(rhs2_strides[0] == 0);
    assert(lhs_strides[2] == 1 && lhs_strides[1] == 0);
#endif

    bool rhs1_transposed;
    bool rhs2_transposed;
    size_t rhs1_stride = stride_for_blas(m, k, rhs1_strides[0], rhs1_strides[1], rhs1_transposed);
    size_t rhs2_stride = stride_for_blas(k, n, rhs2_strides[1], rhs2_strides[2], rhs2_transposed);

    MatMulImplBody<KIND, CODE>()(m,
                                 n,
                                 k,
                                 lhs,
                                 rhs1,
                                 rhs2,
                                 lhs_strides[0],
                                 rhs1_stride,
                                 rhs2_stride,
                                 rhs1_transposed,
                                 rhs2_transposed,
                                 args.lhs.is_readable());
  }

  template <Type::Code CODE, std::enable_if_t<!support_matmul<CODE>::value>* = nullptr>
  void operator()(MatMulArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void matmul_template(TaskContext& context)
{
  auto& reductions = context.reductions();
  auto& inputs     = context.inputs();

  MatMulArgs args{reductions[0], inputs[0], inputs[1]};
  // Note that we can't dispatch on the lhs's type,
  // as the lhs can have a different type than the rhs'
  type_dispatch(args.rhs1.code(), MatMulImpl<KIND>{}, args);
}

}  // namespace cunumeric
