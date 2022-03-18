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

#include "cunumeric/index/zip.h"
#include "cunumeric/index/zip_template.inl"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <int DIM, int N>
struct ZipImplBody<VariantKind::CPU, DIM, N> {
  using VAL = int64_t;

  template <size_t... Is>
  void operator()(const AccessorWO<Point<N>, DIM>& out,
                  const std::vector<AccessorRO<VAL, DIM>>& index_arrays,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  bool dense,
                  const int64_t key_dim,
                  const int64_t start_index,
                  std::index_sequence<Is...>) const
  {
    if (index_arrays.size() == N) {
      const size_t volume = rect.volume();
      if (dense) {
        auto outptr = out.ptr(rect);
        for (size_t idx = 0; idx < volume; ++idx) {
          outptr[idx] = Legion::Point<N>(index_arrays[Is].ptr(rect)[idx]...);
        }
      } else {
        for (size_t idx = 0; idx < volume; ++idx) {
          auto p = pitches.unflatten(idx, rect.lo);
          out[p] = Legion::Point<N>(index_arrays[Is][p]...);
        }
      }
    } else if (index_arrays.size() < N) {
      const size_t volume = rect.volume();
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        Legion::Point<N> new_point;
        for (size_t i = 0; i < start_index; i++) { new_point[i] = p[i]; }
        for (size_t i = 0; i < index_arrays.size(); i++) {
          new_point[start_index + i] = index_arrays[i][p];
        }
        for (size_t i = (start_index + index_arrays.size()); i < N; i++) {
          int64_t j    = key_dim + i - 1 - (index_arrays.size() - 1);
          new_point[i] = p[j];
        }
        out[p] = new_point;
      }
    }
  }
};

/*static*/ void ZipTask::cpu_variant(TaskContext& context)
{
  zip_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { ZipTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
