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

using namespace Legion;
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
  void operator()(FFTArgs& args, std::vector<comm::Communicator>& comms) const
  {
    using INPUT_TYPE  = legate_type_of<CODE_IN>;
    using OUTPUT_TYPE = legate_type_of<FFT<FFT_TYPE, CODE_IN>::CODE_OUT>;

    auto in_rect  = args.input.shape<DIM>();
    auto out_rect = args.output.shape<DIM>();
    if (in_rect.empty() || out_rect.empty()) return;

    auto input  = args.input.read_accessor<INPUT_TYPE, DIM>(in_rect);
    auto output = args.output.write_accessor<OUTPUT_TYPE, DIM>(out_rect);

    FFTImplBody<KIND, FFT_TYPE, FFT<FFT_TYPE, CODE_IN>::CODE_OUT, CODE_IN, DIM>()(
      output,
      input,
      out_rect,
      in_rect,
      args.axes,
      args.direction,
      args.operate_over_axes,
      args.gpu_id,
      args.num_gpus,
      comms);
  }

  // We only support up to 3D FFTs for now
  template <LegateTypeCode CODE_IN,
            int32_t DIM,
            std::enable_if_t<((DIM > 3) || !FFT<FFT_TYPE, CODE_IN>::valid)>* = nullptr>
  void operator()(FFTArgs& args, std::vector<comm::Communicator>& comms) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct FFTDispatch {
  template <CuNumericFFTType FFT_TYPE>
  void operator()(FFTArgs& args, std::vector<comm::Communicator>& comms) const
  {
    // Not expecting changing dimensions, at least for now
    assert(args.input.dim() == args.output.dim());

    double_dispatch(args.input.dim(), args.input.code(), FFTImpl<KIND, FFT_TYPE>{}, args, comms);
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
  int arg_idx            = 0;
  args.type              = scalars[arg_idx++].value<CuNumericFFTType>();
  args.direction         = scalars[arg_idx++].value<CuNumericFFTDirection>();
  args.num_gpus          = scalars[arg_idx++].value<int32_t>();
  args.operate_over_axes = scalars[arg_idx++].value<bool>();

  while (arg_idx < scalars.size()) args.axes.push_back(scalars[arg_idx++].value<int64_t>());

  args.gpu_id = context.get_task_index()[0];

  // some sanity checks
  // We assume that all gpus within a task are on the same node
  // AND the id corresponds to their natural order
  // This needs to be ensured by the resource scoping
  assert(context.is_single_task() || args.num_gpus > 1);
  if (args.num_gpus > 1) {
    auto domain = context.get_launch_domain();
    assert(args.input.dim() == 1);
    assert(domain.get_dim() == 1);
    assert(args.num_gpus == domain.hi()[0] - domain.lo()[0] + 1);
  }

  fft_dispatch(args.type, FFTDispatch<KIND>{}, args, context.communicators());
}
}  // namespace cunumeric
