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

#include "unary/scalar_unary_red.h"
#include "unary/scalar_unary_red_template.inl"

#include <omp.h>

namespace legate {
namespace numpy {

using namespace Legion;

template <UnaryRedCode OP_CODE, LegateTypeCode CODE, int DIM>
struct ScalarUnaryRedImplBody<VariantKind::OMP, OP_CODE, CODE, DIM> {
  using OP    = UnaryRedOp<OP_CODE, CODE>;
  using LG_OP = typename OP::OP;
  using VAL   = legate_type_of<CODE>;

  void operator()(OP func,
                  AccessorRD<LG_OP, true, 1> out,
                  AccessorRO<VAL, DIM> in,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  bool dense) const
  {
    auto result            = LG_OP::identity;
    const size_t volume    = rect.volume();
    const auto max_threads = omp_get_max_threads();
    auto locals            = static_cast<VAL*>(alloca(max_threads * sizeof(VAL)));
    for (auto idx = 0; idx < max_threads; ++idx) locals[idx] = LG_OP::identity;
    if (dense) {
      auto inptr = in.ptr(rect);
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (size_t idx = 0; idx < volume; ++idx) OP::template fold<true>(locals[tid], inptr[idx]);
      }
    } else {
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (size_t idx = 0; idx < volume; ++idx) {
          auto p = pitches.unflatten(idx, rect.lo);
          OP::template fold<true>(locals[tid], in[p]);
        }
      }
    }

    for (auto idx = 0; idx < max_threads; ++idx) out.reduce(0, locals[idx]);
  }
};

template <LegateTypeCode CODE, int DIM>
struct ScalarUnaryRedImplBody<VariantKind::OMP, UnaryRedCode::CONTAINS, CODE, DIM> {
  using OP    = UnaryRedOp<UnaryRedCode::SUM, LegateTypeCode::BOOL_LT>;
  using LG_OP = typename OP::OP;
  using VAL   = legate_type_of<CODE>;

  void operator()(AccessorRD<LG_OP, true, 1> out,
                  AccessorRO<VAL, DIM> in,
                  const Store& to_find_scalar,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  bool dense) const
  {
    auto result            = LG_OP::identity;
    const auto to_find     = to_find_scalar.scalar<VAL>();
    const size_t volume    = rect.volume();
    const auto max_threads = omp_get_max_threads();
    auto locals            = static_cast<bool*>(alloca(max_threads * sizeof(VAL)));
    for (auto idx = 0; idx < max_threads; ++idx) locals[idx] = false;
    if (dense) {
      auto inptr = in.ptr(rect);
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (size_t idx = 0; idx < volume; ++idx)
          if (inptr[idx] == to_find) locals[tid] = true;
      }
    } else {
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (size_t idx = 0; idx < volume; ++idx) {
          auto point = pitches.unflatten(idx, rect.lo);
          if (in[point] == to_find) locals[tid] = true;
        }
      }
    }

    for (auto idx = 0; idx < max_threads; ++idx) out.reduce(0, locals[idx]);
  }
};

template <LegateTypeCode CODE, int DIM>
struct ScalarUnaryRedImplBody<VariantKind::OMP, UnaryRedCode::COUNT_NONZERO, CODE, DIM> {
  using OP    = UnaryRedOp<UnaryRedCode::SUM, LegateTypeCode::UINT64_LT>;
  using LG_OP = typename OP::OP;
  using VAL   = legate_type_of<CODE>;

  void operator()(AccessorRD<LG_OP, true, 1> out,
                  AccessorRO<VAL, DIM> in,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  bool dense) const
  {
    auto result            = LG_OP::identity;
    const size_t volume    = rect.volume();
    const auto max_threads = omp_get_max_threads();
    auto locals            = static_cast<uint64_t*>(alloca(max_threads * sizeof(VAL)));
    for (auto idx = 0; idx < max_threads; ++idx) locals[idx] = 0;
    if (dense) {
      auto inptr = in.ptr(rect);
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (size_t idx = 0; idx < volume; ++idx) locals[tid] += inptr[idx] != VAL(0);
      }
    } else {
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (size_t idx = 0; idx < volume; ++idx) {
          auto point = pitches.unflatten(idx, rect.lo);
          locals[tid] += in[point] != VAL(0);
        }
      }
    }

    for (auto idx = 0; idx < max_threads; ++idx) out.reduce(0, locals[idx]);
  }
};

/*static*/ void ScalarUnaryRedTask::omp_variant(TaskContext& context)
{
  scalar_unary_red_template<VariantKind::OMP>(context);
}

}  // namespace numpy
}  // namespace legate
