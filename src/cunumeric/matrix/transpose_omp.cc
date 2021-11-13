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

#include "cunumeric/matrix/transpose.h"
#include "cunumeric/matrix/transpose_template.inl"

#include "omp.h"
#include "cblas.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE>
struct TransposeImplBody<VariantKind::OMP, CODE, 2> {
  using _VAL = legate_type_of<CODE>;

  template <typename VAL = _VAL, std::enable_if_t<sizeof(VAL) == 4>* = nullptr>
  void operator()(const Rect<2>& out_rect,
                  const Rect<2>& in_rect,
                  const AccessorWO<VAL, 2>& out,
                  const AccessorRO<VAL, 2>& in) const
  {
    size_t out_strides[2];
    size_t in_strides[2];
    auto out_ptr      = reinterpret_cast<float*>(out.ptr(out_rect, out_strides));
    const auto in_ptr = reinterpret_cast<const float*>(in.ptr(in_rect, in_strides));
    const coord_t m   = (in_rect.hi[0] - in_rect.lo[0]) + 1;
    const coord_t n   = (in_rect.hi[1] - in_rect.lo[1]) + 1;
    cblas_somatcopy(CblasRowMajor,
                    CblasTrans,
                    m,
                    n,
                    1.f /*scale*/,
                    in_ptr,
                    in_strides[0],
                    out_ptr,
                    out_strides[0]);
  }

  template <typename VAL = _VAL, std::enable_if_t<sizeof(VAL) == 8>* = nullptr>
  void operator()(const Rect<2>& out_rect,
                  const Rect<2>& in_rect,
                  const AccessorWO<VAL, 2>& out,
                  const AccessorRO<VAL, 2>& in) const
  {
    size_t out_strides[2];
    size_t in_strides[2];
    auto out_ptr      = reinterpret_cast<double*>(out.ptr(out_rect, out_strides));
    const auto in_ptr = reinterpret_cast<const double*>(in.ptr(in_rect, in_strides));
    const coord_t m   = (in_rect.hi[0] - in_rect.lo[0]) + 1;
    const coord_t n   = (in_rect.hi[1] - in_rect.lo[1]) + 1;
    cblas_domatcopy(CblasRowMajor,
                    CblasTrans,
                    m,
                    n,
                    1.f /*scale*/,
                    in_ptr,
                    in_strides[0],
                    out_ptr,
                    out_strides[0]);
  }

  template <typename VAL = _VAL, std::enable_if_t<sizeof(VAL) != 4 && sizeof(VAL) != 8>* = nullptr>
  void operator()(const Rect<2>& out_rect,
                  const Rect<2>& in_rect,
                  const AccessorWO<VAL, 2>& out,
                  const AccessorRO<VAL, 2>& in) const
  {
    constexpr coord_t BF = 128 / sizeof(VAL);
#pragma omp parallel for
    for (coord_t i1 = in_rect.lo[0]; i1 <= in_rect.hi[0]; i1 += BF) {
      for (coord_t j1 = in_rect.lo[1]; j1 <= in_rect.hi[1]; j1 += BF) {
        const coord_t max_i2 = ((i1 + BF) <= in_rect.hi[0]) ? i1 + BF : in_rect.hi[0];
        const coord_t max_j2 = ((j1 + BF) <= in_rect.hi[1]) ? j1 + BF : in_rect.hi[1];
        for (int i2 = i1; i2 <= max_i2; i2++)
          for (int j2 = j1; j2 <= max_j2; j2++) out[j2][i2] = in[i2][j2];
      }
    }
  }
};

/*static*/ void TransposeTask::omp_variant(TaskContext& context)
{
  transpose_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
