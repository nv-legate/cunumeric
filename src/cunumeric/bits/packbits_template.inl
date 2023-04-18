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
#include "cunumeric/bits/packbits.h"
#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, Type CODE, int32_t DIM, Bitorder BITORDER>
struct PackbitsImplBody;

template <VariantKind KIND, Bitorder BITORDER>
struct PackbitsImpl {
  template <Type CODE, int32_t DIM, std::enable_if_t<is_integral<CODE>::value>* = nullptr>
  void operator()(Array& output, Array& input, uint32_t axis) const
  {
    using VAL = legate_type_of<CODE>;

    auto out_rect = output.shape<DIM>();

    if (out_rect.empty()) return;

    auto in_rect = input.shape<DIM>();

    auto out = output.write_accessor<uint8_t, DIM>(out_rect);
    auto in  = input.read_accessor<VAL, DIM>(in_rect);

    // Compute an output rectangle where each output element can use all 8 input elements
    // for packing
    auto aligned_rect     = out_rect;
    int64_t axis_extent   = in_rect.hi[axis] - in_rect.lo[axis] + 1;
    aligned_rect.hi[axis] = aligned_rect.lo[axis] + axis_extent / 8 - 1;
#ifdef DEBUG_CUNUMERIC
    assert(aligned_rect.hi[axis] <= out_rect.hi[axis]);
#endif

    auto unaligned_rect     = out_rect;
    unaligned_rect.lo[axis] = aligned_rect.hi[axis] + 1;
#ifdef DEBUG_CUNUMERIC
    assert(unaligned_rect.union_bbox(aligned_rect) == out_rect);
#endif

    Pitches<DIM - 1> aligned_pitches, unaligned_pitches;
    auto aligned_volume   = aligned_pitches.flatten(aligned_rect);
    auto unaligned_volume = unaligned_pitches.flatten(unaligned_rect);

    PackbitsImplBody<KIND, CODE, DIM, BITORDER>{}(out,
                                                  in,
                                                  aligned_rect,
                                                  unaligned_rect,
                                                  aligned_pitches,
                                                  unaligned_pitches,
                                                  aligned_volume,
                                                  unaligned_volume,
                                                  in_rect.hi[axis],
                                                  axis);
  }

  template <Type CODE, int32_t DIM, std::enable_if_t<!is_integral<CODE>::value>* = nullptr>
  void operator()(Array& output, Array& input, uint32_t axis) const
  {
    // Unreachable
    assert(false);
  }
};

template <VariantKind KIND>
static void packbits_template(TaskContext& context)
{
  auto& output  = context.outputs().front();
  auto& input   = context.inputs().front();
  auto& scalars = context.scalars();
  auto axis     = scalars[0].value<uint32_t>();
  auto bitorder = scalars[1].value<Bitorder>();

  auto code = input.code<Type>();
  switch (bitorder) {
    case Bitorder::BIG: {
      double_dispatch(input.dim(), code, PackbitsImpl<KIND, Bitorder::BIG>{}, output, input, axis);
      break;
    }
    case Bitorder::LITTLE: {
      double_dispatch(
        input.dim(), code, PackbitsImpl<KIND, Bitorder::LITTLE>{}, output, input, axis);
      break;
    }
  }
}

}  // namespace cunumeric
