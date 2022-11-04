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

template <VariantKind KIND, LegateTypeCode CODE, int DIM, int VDIM, bool SCALAR_VALUE = false>
struct Putmask {
  using T      = legate_type_of<CODE>;
  using IN     = AccessorRW<T, DIM>;
  using MASK   = AccessorRO<bool, DIM>;
  using VALUES = AccessorRO<T, VDIM>;

  IN input;
  T* inputptr;
  MASK mask;
  const bool* maskptr;
  VALUES values;
  const T* valptr;
  Pitches<DIM - 1> pitches;
  Rect<DIM> rect;
  Rect<VDIM> vrect;
  bool dense;

  struct DenseTag {};
  struct SparseTag {};

  // constructor:
  Putmask(PutmaskArgs& args) : dense(false)
  {
    rect  = args.input.shape<DIM>();
    vrect = args.values.shape<VDIM>();
#ifdef DEBUG_CUNUMERIC
    if constexpr (SCALAR_VALUE)
      assert(rect == args.mask.shape<DIM>());
    else
      assert((rect == args.mask.shape<DIM>()) && (rect == vrect));
#endif

    input  = args.input.read_write_accessor<T, DIM>(rect);
    mask   = args.mask.read_accessor<bool, DIM>(rect);
    values = args.values.read_accessor<T, VDIM>(vrect);
#ifndef LEGION_BOUNDS_CHECKS
    dense = input.accessor.is_dense_row_major(rect) && mask.accessor.is_dense_row_major(rect);
    if constexpr (!SCALAR_VALUE) dense = dense && values.accessor.is_dense_row_major(rect);
    if (dense) {
      inputptr = input.ptr(rect);
      maskptr  = mask.ptr(rect);
      if constexpr (!SCALAR_VALUE) valptr = values.ptr(vrect);
    }
#endif
  }  // constructor

  __CUDA_HD__ void operator()(const size_t idx, DenseTag) const noexcept
  {
    if constexpr (SCALAR_VALUE) {
      if (maskptr[idx]) inputptr[idx] = values[0];
    } else {
      if (maskptr[idx]) inputptr[idx] = valptr[idx];
    }
  }

  __CUDA_HD__ void operator()(const size_t idx, SparseTag) const noexcept
  {
    auto p = pitches.unflatten(idx, rect.lo);
    if constexpr (SCALAR_VALUE) {
      if (mask[p]) input[p] = values[0];
    } else if (mask[p])
      input[p] = values[p];
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
    if (args.is_scalar_value) {
      Putmask<KIND, CODE, DIM, 1, true> putmask(args);
      putmask.execute();
    } else {
      Putmask<KIND, CODE, DIM, DIM> putmask(args);
      putmask.execute();
    }
  }
};

template <VariantKind KIND>
static void putmask_template(TaskContext& context)
{
  auto& inputs         = context.inputs();
  bool is_scalar_value = context.scalars()[0].value<bool>();
  PutmaskArgs args{inputs[0], inputs[1], inputs[2], is_scalar_value};
  double_dispatch(args.input.dim(), args.input.code(), PutmaskImpl<KIND>{}, args);
}

}  // namespace cunumeric
