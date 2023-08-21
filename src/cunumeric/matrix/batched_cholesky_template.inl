/* Copyright 2023 NVIDIA Corporation
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
#include <core/task/exception.h>
#include "cunumeric/cunumeric.h"
#include "cunumeric/matrix/batched_cholesky.h"
#include "cunumeric/matrix/potrf_template.inl"
#include "cunumeric/matrix/transpose_template.inl"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct BatchedCholeskyImplBody {
  template <class T>
  void operator()(T* array, int32_t m, int32_t n)
  {
    PotrfImplBody<KIND, CODE>()(array, m, n);
  }
};

template <VariantKind KIND>
struct CopyBlockImpl {
  void operator()(void* dst, const void* src, size_t n);
};

template <VariantKind KIND, Type::Code CODE>
struct BatchedTransposeImplBody {
  using VAL = legate_type_of<CODE>;

  void operator()(VAL* array, int32_t n);
};

template <Type::Code CODE>
struct _cholesky_supported {
  static constexpr bool value = CODE == Type::Code::FLOAT64 || CODE == Type::Code::FLOAT32 ||
                                CODE == Type::Code::COMPLEX64 || CODE == Type::Code::COMPLEX128;
};

template <VariantKind KIND>
struct BatchedCholeskyImpl {
  template <Type::Code CODE, int DIM>
  void operator()(Array& input_array, Array& output_array) const
  {
    using VAL = legate_type_of<CODE>;

    auto shape = input_array.shape<DIM>();
    if (shape != output_array.shape<DIM>()) {
      throw legate::TaskException(
        "Batched cholesky is not supported when input/output shapes differ");
    }

    auto ncols = shape.hi[DIM - 1] - shape.lo[DIM - 1] + 1;

    size_t in_strides[DIM];
    size_t out_strides[DIM];

    auto input = input_array.read_accessor<VAL, DIM>(shape).ptr(shape, in_strides);
    if (in_strides[DIM - 2] != ncols || in_strides[DIM - 1] != 1) {
      throw legate::TaskException(
        "Bad input accessor in batched cholesky, last two dimensions must be non-transformed and "
        "dense with stride == 1");
    }

    auto output = output_array.write_accessor<VAL, DIM>(shape).ptr(shape, out_strides);
    if (out_strides[DIM - 2] != ncols || out_strides[DIM - 1] != 1) {
      throw legate::TaskException(
        "Bad output accessor in batched cholesky, last two dimensions must be non-transformed and "
        "dense with stride == 1");
    }

    if (shape.empty()) return;

    // TODO: we need some sort of check here on the strides
    // This should be a dense thing.

    int num_blocks = 1;
    for (int i = 0; i < (DIM - 2); ++i) { num_blocks *= (shape.hi[i] - shape.lo[i] + 1); }

    auto m = static_cast<int32_t>(shape.hi[DIM - 2] - shape.lo[DIM - 2] + 1);
    auto n = static_cast<int32_t>(shape.hi[DIM - 1] - shape.lo[DIM - 1] + 1);
    assert(m > 0 && n > 0);

    auto block_stride = m * n;

    for (int i = 0; i < num_blocks; ++i) {
      if constexpr (_cholesky_supported<CODE>::value) {
        CopyBlockImpl<KIND>()(output, input, sizeof(VAL) * block_stride);
        PotrfImplBody<KIND, CODE>()(output, m, n);
        // Implicit assumption here about the cholesky code created.
        // We assume the output has C layout, but each subblock
        // will be generated in Fortran layout. Transpose the Fortran
        // subblock into C layout.
        // CHANGE: If this code is changed, please make sure all changes
        // are consistent with those found in mapper.cc.
        BatchedTransposeImplBody<KIND, CODE>()(output, n);
        input += block_stride;
        output += block_stride;
      }
    }
  }
};

template <VariantKind KIND>
static void batched_cholesky_task_context_dispatch(TaskContext& context)
{
  auto& batched_input  = context.inputs()[0];
  auto& batched_output = context.outputs()[0];
  if (batched_input.code() != batched_output.code()) {
    throw legate::TaskException(
      "batched cholesky is not yet supported when input/output types differ");
  }
  if (batched_input.dim() != batched_output.dim()) {
    throw legate::TaskException("input/output have different dims in batched cholesky");
  }
  if (batched_input.dim() <= 2) {
    throw legate::TaskException(
      "internal error: batched cholesky input does not have more than 2 dims");
  }
  double_dispatch(batched_input.dim(),
                  batched_input.code(),
                  BatchedCholeskyImpl<KIND>{},
                  batched_input,
                  batched_output);
}

}  // namespace cunumeric
