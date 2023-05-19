/* Copyright 2022 NVIDIA Corporation
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

#include "cunumeric/scan/scan_local_util.h"
#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, ScanCode OP_CODE, Type::Code CODE, int DIM>
struct ScanLocalImplBody;

template <VariantKind KIND, ScanCode OP_CODE, Type::Code CODE, int DIM>
struct ScanLocalNanImplBody;

template <VariantKind KIND, ScanCode OP_CODE, bool NAN_TO_IDENTITY>
struct ScanLocalImpl {
  // Case where NANs are transformed
  template <Type::Code CODE,
            int DIM,
            std::enable_if_t<NAN_TO_IDENTITY && (legate::is_floating_point<CODE>::value ||
                                                 legate::is_complex<CODE>::value)>* = nullptr>
  void operator()(ScanLocalArgs& args) const
  {
    using OP  = ScanOp<OP_CODE, CODE>;
    using VAL = legate_type_of<CODE>;

    auto rect = args.out.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) {
      args.sum_vals.bind_empty_data();
      return;
    }

    auto out = args.out.write_accessor<VAL, DIM>(rect);
    auto in  = args.in.read_accessor<VAL, DIM>(rect);

    OP func;
    ScanLocalNanImplBody<KIND, OP_CODE, CODE, DIM>()(func, out, in, args.sum_vals, pitches, rect);
  }
  // Case where NANs are as is
  template <Type::Code CODE,
            int DIM,
            std::enable_if_t<!(NAN_TO_IDENTITY && (legate::is_floating_point<CODE>::value ||
                                                   legate::is_complex<CODE>::value))>* = nullptr>
  void operator()(ScanLocalArgs& args) const
  {
    using OP  = ScanOp<OP_CODE, CODE>;
    using VAL = legate_type_of<CODE>;

    auto rect = args.out.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) {
      args.sum_vals.bind_empty_data();
      return;
    }

    auto out = args.out.write_accessor<VAL, DIM>(rect);
    auto in  = args.in.read_accessor<VAL, DIM>(rect);

    OP func;
    ScanLocalImplBody<KIND, OP_CODE, CODE, DIM>()(func, out, in, args.sum_vals, pitches, rect);
  }
};

template <VariantKind KIND>
struct ScanLocalDispatch {
  template <ScanCode OP_CODE, bool NAN_TO_IDENTITY>
  void operator()(ScanLocalArgs& args) const
  {
    return double_dispatch(
      args.in.dim(), args.in.code(), ScanLocalImpl<KIND, OP_CODE, NAN_TO_IDENTITY>{}, args);
  }
};

template <VariantKind KIND>
static void scan_local_template(TaskContext& context)
{
  ScanLocalArgs args{context.outputs()[0],
                     context.inputs()[0],
                     context.outputs()[1],
                     context.scalars()[0].value<ScanCode>(),
                     context.scalars()[1].value<bool>()};
  op_dispatch(args.op_code, args.nan_to_identity, ScanLocalDispatch<KIND>{}, args);
}

}  // namespace cunumeric
