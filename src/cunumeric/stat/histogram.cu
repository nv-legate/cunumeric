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

#include "cunumeric/stat/histogram.cuh"
#include "cunumeric/stat/histogram_impl.h"

#include "cunumeric/stat/histogram.h"
#include "cunumeric/stat/histogram_template.inl"

#include "cunumeric/cuda_help.h"

namespace cunumeric {
template <Type::Code CODE>
struct HistogramImplBody<VariantKind::GPU, CODE> {
  using VAL = legate_type_of<CODE>;

  // for now, it has been decided to hardcode these types:
  //
  using BinType    = double;
  using WeightType = double;

  // in the future we might relax relax that requirement,
  // but complicate dispatching:
  //
  // template <typename BinType = VAL, typename WeightType = VAL>
  void operator()(const AccessorRO<VAL, 1>& src,
                  const Rect<1>& src_rect,
                  const AccessorRO<BinType, 1>& bins,
                  const Rect<1>& bins_rect,
                  const AccessorRO<WeightType, 1>& weights,
                  const Rect<1>& weights_rect,
                  const AccessorRD<SumReduction<WeightType>, true, 1>& result,
                  const Rect<1>& result_rect) const
  {
    auto stream = get_cached_stream();

    size_t src_strides[1];
    auto src_rect      = args.src.shape<1>();
    auto src_acc       = args.src.read_accessor<VAL, 1>(src_rect);
    const VAL* src_ptr = src_acc.ptr(src_rect, src_strides);
    assert(src_strides[0] == 1);
    //
    // const VAL* src_ptr: need to create a copy with create_buffer(...);
    // since src will get sorted (in-place);
    //
    size_t src_size       = src_rect.hi - src_rec.lo + 1;
    Buffer<VAL*> src_copy = create_buffer<VAL>(src_size, Legion::Memory::Kind::GPU_FB_MEM);
    CHECK_CUDA(
      cudaMemcpyAsync(src_copy.ptr(0), src_ptr, src_size, cudaMemcpyDeviceToDevice, stream));

    // ... at the end of extracting / creating copies:
    //
    CHECK_CUDA(cudaStreamSynchronize(stream));
    // TODO:
    //
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void HistogramTask::gpu_variant(TaskContext& context)
{
  bincount_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
