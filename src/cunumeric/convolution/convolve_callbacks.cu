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

#include "cunumeric/convolution/convolve_common.h"

#include "cunumeric/cuda_help.h"
#include <cufftXt.h>

using namespace Legion;

namespace cunumeric {

template <typename T>
static __device__ T cb_zero_pad(void* data, size_t offset, void* callerinfo, void* sharedptr)
{
  auto* info    = static_cast<const ZeroPadLoadData*>(callerinfo);
  size_t actual = 0;
#pragma unroll 3
  for (int32_t d = 0; d < info->dim; d++) {
    coord_t coord = info->pitches[d].divmod(offset, offset);
    if (coord >= info->bounds[d]) return 0.f;
    actual += coord * info->strides[d];
  }
  auto* ptr = static_cast<T*>(data);
  return ptr[actual - info->misalignment];
}

template <typename Complex, typename Field, Complex (*CTOR)(Field, Field)>
static __device__ Complex cb_multiply(void* data, size_t offset, void* callerinfo, void* sharedptr)
{
  auto* info = static_cast<const LoadComplexData*>(callerinfo);
  auto* ptr  = static_cast<Complex*>(data);
  auto lhs   = ptr[offset];
  auto rhs   = ptr[offset + info->buffervolume];
  return CTOR(lhs.x * rhs.x - lhs.y * rhs.y, lhs.x * rhs.y + lhs.y * rhs.x);
}

template <typename T>
static __device__ void cb_store(
  void* data, size_t offset, T value, void* callerinfo, void* sharedptr)
{
  auto* info    = static_cast<const StoreOutputData<T>*>(callerinfo);
  size_t actual = 0;
#pragma unroll 3
  for (int32_t d = 0; d < info->dim; d++) {
    coord_t coord = info->pitches[d].divmod(offset, offset);
    coord -= info->offsets[d];
    if (coord < 0) return;
    if (coord >= info->bounds[d]) return;
    actual += coord * info->strides[d];
  }
  auto* ptr   = static_cast<T*>(data);
  ptr[actual] = value * info->scale_factor;
}

static __device__ cufftCallbackLoadR d_zero_pad_float  = cb_zero_pad<cufftReal>;
static __device__ cufftCallbackLoadD d_zero_pad_double = cb_zero_pad<cufftDoubleReal>;
static __device__ cufftCallbackLoadC d_multiply_float =
  cb_multiply<cufftComplex, float, make_cuComplex>;
static __device__ cufftCallbackLoadZ d_multiply_double =
  cb_multiply<cufftDoubleComplex, double, make_cuDoubleComplex>;
static __device__ cufftCallbackStoreR d_store_float  = cb_store<cufftReal>;
static __device__ cufftCallbackStoreD d_store_double = cb_store<cufftDoubleReal>;

#define LOAD_SYMBOL(FUN, DEV_FUN)                                 \
  __host__ void* FUN()                                            \
  {                                                               \
    void* ptr = nullptr;                                          \
    CHECK_CUDA(cudaMemcpyFromSymbol(&ptr, DEV_FUN, sizeof(ptr))); \
    assert(ptr != nullptr);                                       \
    return ptr;                                                   \
  }

LOAD_SYMBOL(load_zero_pad_callback_float, d_zero_pad_float)
LOAD_SYMBOL(load_zero_pad_callback_double, d_zero_pad_double)
LOAD_SYMBOL(load_multiply_callback_float, d_multiply_float)
LOAD_SYMBOL(load_multiply_callback_double, d_multiply_double)
LOAD_SYMBOL(load_store_callback_float, d_store_float)
LOAD_SYMBOL(load_store_callback_double, d_store_double)

}  // namespace cunumeric
