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

#include "core/cuda/cuda_help.h"
#include "core/data/buffer.h"

namespace cunumeric {

template <typename REDOP>
class ScalarReductionBuffer {
 private:
  using LHS = typename REDOP::LHS;
  using RHS = typename REDOP::RHS;

 public:
  ScalarReductionBuffer(cudaStream_t stream) : buffer_(legate::create_buffer<LHS>(1))
  {
    // This will prevent this template from getting instantiated at compile time
    // if LHS and RHS are different types
    LHS identity{REDOP::identity};
    ptr_ = buffer_.ptr(0);
    CHECK_CUDA(cudaMemcpyAsync(ptr_, &identity, sizeof(LHS), cudaMemcpyHostToDevice, stream));
  }

  __device__ void operator<<=(const RHS& value) const
  {
    REDOP::template fold<false /*exclusive*/>(*ptr_, value);
  }

  __host__ LHS read(cudaStream_t stream) const
  {
    LHS result{REDOP::identity};
    CHECK_CUDA(cudaMemcpyAsync(&result, ptr_, sizeof(LHS), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return result;
  }

  __device__ LHS read() const { return *ptr_; }

 private:
  legate::Buffer<LHS> buffer_;
  LHS* ptr_;
};

}  // namespace cunumeric
