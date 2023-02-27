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
#include "cunumeric/nullary/window.h"
#include "cunumeric/nullary/window_util.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, WindowOpCode OP_CODE>
struct WindowImplBody;

template <VariantKind KIND>
struct WindowImpl {
  template <WindowOpCode OP_CODE>
  void operator()(Array& output, int64_t M, double beta) const
  {
    auto rect = output.shape<1>();

    if (rect.empty()) return;

    auto out = output.write_accessor<double, 1>(rect);

#ifndef LEGATE_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    WindowImplBody<KIND, OP_CODE>{}(out, rect, dense, M, beta);
  }
};

template <VariantKind KIND>
static void window_template(TaskContext& context)
{
  auto& output  = context.outputs().front();
  auto& scalars = context.scalars();
  auto op_code  = scalars[0].value<WindowOpCode>();
  auto M        = scalars[1].value<int64_t>();
  auto beta     = scalars.size() > 2 ? scalars[2].value<double>() : 0.0;

  op_dispatch(op_code, WindowImpl<KIND>{}, output, M, beta);
}

}  // namespace cunumeric
