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

namespace legate {
namespace numpy {

using namespace Legion;

template <VariantKind KIND, LegateTypeCode CODE>
struct MatMulImplBody;

template <LegateTypeCode CODE>
struct support_matmul : std::false_type {
};
template <>
struct support_matmul<LegateTypeCode::DOUBLE_LT> : std::true_type {
  using ACC_TYPE = double;
};
template <>
struct support_matmul<LegateTypeCode::FLOAT_LT> : std::true_type {
  using ACC_TYPE = float;
};
template <>
struct support_matmul<LegateTypeCode::HALF_LT> : std::true_type {
  using ACC_TYPE = float;
};

template <VariantKind KIND>
struct MatMulImpl {
  template <LegateTypeCode CODE, std::enable_if_t<support_matmul<CODE>::value> * = nullptr>
  void operator()(MatMulArgs &args) const
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

    const auto m = shape.hi[0] - shape.lo[0] + 1;
    const auto k = shape.hi[1] - shape.lo[1] + 1;
    const auto n = shape.hi[2] - shape.lo[2] + 1;

    size_t lhs_strides[3];
    size_t rhs1_strides[3];
    size_t rhs2_strides[3];

    auto rhs1 = args.rhs1.read_accessor<VAL, 3>(shape).ptr(shape, rhs1_strides);
    auto rhs2 = args.rhs2.read_accessor<VAL, 3>(shape).ptr(shape, rhs2_strides);
    auto lhs  = args.lhs.reduce_accessor<SumReduction<ACC>, true, 3>(shape).ptr(shape, lhs_strides);

    auto rhs1_stride = std::max(rhs1_strides[0], rhs1_strides[1]);
    auto rhs2_stride = std::max(rhs2_strides[1], rhs2_strides[2]);
    auto rhs1_transposed =
      (rhs1_strides[0] != rhs1_strides[1]) ? (rhs1_strides[1] == rhs1_stride) : (rhs1_stride != k);
    auto rhs2_transposed =
      (rhs2_strides[1] != rhs2_strides[2]) ? (rhs2_strides[2] == rhs2_stride) : (rhs2_stride != n);

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
                                 rhs2_transposed);
  }

  template <LegateTypeCode CODE, std::enable_if_t<!support_matmul<CODE>::value> * = nullptr>
  void operator()(MatMulArgs &args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void matmul_template(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context context,
                            Runtime *runtime)
{
  Deserializer ctx(task, regions);
  MatMulArgs args;
  deserialize(ctx, args);
  // Note that we can't dispatch on the lhs's type,
  // as the lhs can have a different type than the rhs'
  type_dispatch(args.rhs1.code(), MatMulImpl<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
