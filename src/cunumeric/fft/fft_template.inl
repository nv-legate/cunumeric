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

#pragma once

// Useful for IDEs
#include "cunumeric/fft/fft.h"
#include "cunumeric/pitches.h"
#include "cunumeric/fft/fft_util.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND,
          CuNumericFFTType FFT_TYPE,
          LegateTypeCode CODE_OUT,
          LegateTypeCode CODE_IN,
          int32_t DIM>
struct FFTImplBody;

template <VariantKind KIND, CuNumericFFTType FFT_TYPE>
struct FFTImpl {
  template <LegateTypeCode CODE_IN,
            int32_t DIM,
            std::enable_if_t<((DIM <= 3) && FFT<FFT_TYPE, CODE_IN>::valid)>* = nullptr>
  void operator()(FFTArgs& args) const
  {
    using INPUT_TYPE  = legate_type_of<CODE_IN>;
    using OUTPUT_TYPE = legate_type_of<FFT<FFT_TYPE, CODE_IN>::CODE_OUT>;

    auto in_rect  = args.input.shape<DIM>();
    auto out_rect = args.output.shape<DIM>();
    if (in_rect.empty() || out_rect.empty()) return;

    auto input  = args.input.read_accessor<INPUT_TYPE, DIM>(in_rect);
    auto output = args.output.write_accessor<OUTPUT_TYPE, DIM>(out_rect);

    FFTImplBody<KIND, FFT_TYPE, FFT<FFT_TYPE, CODE_IN>::CODE_OUT, CODE_IN, DIM>()(
      output, input, out_rect, in_rect, args.axes, args.direction, args.operate_over_axes);
  }

  // We only support up to 3D FFTs for now
  template <LegateTypeCode CODE_IN,
            int32_t DIM,
            std::enable_if_t<((DIM > 3) || !FFT<FFT_TYPE, CODE_IN>::valid)>* = nullptr>
  void operator()(FFTArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct FFTDispatch {
  template <CuNumericFFTType FFT_TYPE>
  void operator()(FFTArgs& args) const
  {
    // Not expecting changing dimensions, at least for now
    assert(args.input.dim() == args.output.dim());

    double_dispatch(args.input.dim(), args.input.code(), FFTImpl<KIND, FFT_TYPE>{}, args);
  }
};

template <VariantKind KIND>
static void fft_template(TaskContext& context)
{
  FFTArgs args;

  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();
  auto& scalars = context.scalars();

  args.output = std::move(outputs[0]);
  args.input  = std::move(inputs[0]);
  // Scalar arguments. Pay attention to indexes / ranges when adding or reordering arguments
  args.type              = scalars[0].value<CuNumericFFTType>();
  args.direction         = scalars[1].value<CuNumericFFTDirection>();
  args.operate_over_axes = scalars[2].value<bool>();

  for (size_t i = 3; i < scalars.size(); ++i) args.axes.push_back(scalars[i].value<int64_t>());

  fft_dispatch(args.type, FFTDispatch<KIND>{}, args);
}
}  // namespace cunumeric
