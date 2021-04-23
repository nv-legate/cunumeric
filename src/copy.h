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

#ifndef __NUMPY_COPY_H__
#define __NUMPY_COPY_H__

#include "unary_operation.h"

// This is a basic copy task that knows how to copy data
// from one region to another with transforms. We should
// be able to remove the need for this once we have scatter
// and gather copies from Realm and Legion.

namespace legate {
namespace numpy {
template <class T>
struct CopyOperation {
  using argument_type           = T;
  constexpr static auto op_code = NumPyOpCode::NUMPY_COPY;

  __CUDA_HD__ constexpr T operator()(const T& x) const { return x; }
};

template <class T>
class CopyTask : public UnaryOperationTask<CopyTask<T>, CopyOperation<T>> {
};
}  // namespace numpy
}  // namespace legate

#endif  // __NUMPY_COPY_H__
