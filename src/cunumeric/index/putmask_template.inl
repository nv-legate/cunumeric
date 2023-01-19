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
#include <core/utilities/typedefs.h>
#include "cunumeric/index/putmask.h"
#include "cunumeric/pitches.h"
#include "cunumeric/execution_policy/indexing/parallel_loop.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, LegateTypeCode CODE, int DIM>
struct Putmask {
  using T      = legate_type_of<CODE>;
  using IN     = AccessorRW<T, DIM>;
  using MASK   = AccessorRO<bool, DIM>;
  using VALUES = AccessorRO<T, DIM>;

  IN input;
  T* inputptr;
  MASK mask;
  const bool* maskptr;
  VALUES values;
  const T* valptr;
  Pitches<DIM - 1> pitches;
  Rect<DIM> rect;
  bool dense;
  size_t volume;

  struct DenseTag {};
  struct SparseTag {};

  // constructor:
  Putmask(PutmaskArgs& args) : dense(false)
  {
    rect = args.input.shape<DIM>();

    input  = args.input.read_write_accessor<T, DIM>(rect);
    mask   = args.mask.read_accessor<bool, DIM>(rect);
    values = args.values.read_accessor<T, DIM>(rect);
    volume = pitches.flatten(rect);
    if (volume == 0) return;
#ifndef LEGION_BOUNDS_CHECKS
    dense = input.accessor.is_dense_row_major(rect) && mask.accessor.is_dense_row_major(rect);
    dense = dense && values.accessor.is_dense_row_major(rect);
    if (dense) {
      inputptr = input.ptr(rect);
      maskptr  = mask.ptr(rect);
      valptr   = values.ptr(rect);
    }
#endif
  }  // constructor

  __CUDA_HD__ void operator()(const size_t idx, DenseTag) const noexcept
  {
    if (maskptr[idx]) inputptr[idx] = valptr[idx];
  }

  __CUDA_HD__ void operator()(const size_t idx, SparseTag) const noexcept
  {
    auto p = pitches.unflatten(idx, rect.lo);
    if (mask[p]) input[p] = values[p];
  }

  void execute() const noexcept
  {
#ifndef LEGION_BOUNDS_CHECKS
    if (dense) { return ParallelLoopPolicy<KIND, DenseTag>()(rect, *this); }
#endif
    return ParallelLoopPolicy<KIND, SparseTag>()(rect, *this);
  }
};

using namespace Legion;
using namespace legate;

template <VariantKind KIND>
struct PutmaskImpl {
  template <LegateTypeCode CODE, int DIM>
  void operator()(PutmaskArgs& args) const
  {
    Putmask<KIND, CODE, DIM> putmask(args);
    putmask.execute();
  }
};

template <VariantKind KIND>
static void putmask_template(TaskContext& context)
{
  auto& inputs = context.inputs();
  PutmaskArgs args{context.outputs()[0], inputs[1], inputs[2]};
  double_dispatch(args.input.dim(), args.input.code(), PutmaskImpl<KIND>{}, args);
}

}  // namespace cunumeric
