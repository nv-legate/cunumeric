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

#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, int DIM, int N>
struct ZipImplBody;

template <VariantKind KIND>
struct ZipImpl {
  template <int DIM, int N>
  void operator()(ZipArgs& args) const
  {
    using VAL       = int64_t;
    auto out_rect   = args.out.shape<DIM>();
    auto out        = args.out.write_accessor<Point<N>, DIM>(out_rect);
    auto index_rect = args.inputs[0].shape<DIM>();
    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(out_rect);
    if (volume == 0) return;

#ifdef CUNUMERIC_DEBUG
    assert(out_rect == index_rect)
#endif

#ifndef LEGION_BOUNDS_CHECKS
      bool dense = out.accessor.is_dense_row_major(out_rect);
#endif
    std::vector<AccessorRO<VAL, DIM>> index_arrays;
    for (int i = 0; i < args.inputs.size(); i++) {
#ifdef CUNUMERIC_DEBUG
      assert(index_rect == args.inputs[i].shape<DIM>());
#endif
      index_arrays.push_back(args.inputs[i].read_accessor<VAL, DIM>(index_rect));
      dense = dense && index_arrays[i].accessor.is_dense_row_major(out_rect);
    }

#ifdef LEGION_BOUNDS_CHECKS
    bool dense = false;
#endif
    ZipImplBody<KIND, DIM, N>()(out,
                                index_arrays,
                                index_rect,
                                pitches,
                                dense,
                                args.key_dim,
                                args.start_index,
                                std::make_index_sequence<N>());
  }
};

template <VariantKind KIND>
static void zip_template(TaskContext& context)
{
  int64_t N           = context.scalars()[0].value<int64_t>();
  int64_t key_dim     = context.scalars()[1].value<int64_t>();
  int64_t start_index = context.scalars()[2].value<int64_t>();
  ZipArgs args{context.outputs()[0], context.inputs(), N, key_dim, start_index};
  double_dispatch(args.inputs[0].dim(), N, ZipImpl<KIND>{}, args);
}

}  // namespace cunumeric
