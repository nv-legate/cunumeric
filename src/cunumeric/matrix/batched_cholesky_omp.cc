/* Copyright 2021-2022 NVIDIA Corporation
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

#include "cunumeric/matrix/batched_cholesky.h"
#include "cunumeric/matrix/batched_cholesky_template.inl"

#include <cblas.h>
#include <lapack.h>
#include <omp.h>

namespace cunumeric {

using namespace legate;

template <Type::Code CODE>
struct BatchedTransposeImplBody<VariantKind::OMP, CODE> {
  using VAL = legate_type_of<CODE>;

  static constexpr int tile_size = 64;

  void operator()(VAL* out, int n) const
  {

    int nblocks = (n + tile_size - 1) / tile_size;

#pragma omp parallel for
    for (int rb=0; rb < nblocks; ++rb){
      for (int cb=0; cb < nblocks; ++cb){
        VAL tile[tile_size][tile_size];
        int r_start = rb * tile_size;
        int r_stop = std::min(r_start + tile_size, n);
        int c_start = cb * tile_size;
        int c_stop = std::min(c_start + tile_size, n);

        for (int r=r_start, tr=0; r < r_stop; ++r){
          for (int c=c_start, tc=0; c < c_stop; ++c){
            if (r <= c){
              tile[tr][tc] = out[r*n + c];
            } else {
              tile[tr][tc] = 0;
            }
          }
        }

        for (int r=c_start, tr=0; r < c_stop; ++r){
          for (int c=r_start, tc=0; c < r_stop; ++c){
            out[r*n+c] = tile[tr][tc];
          }
        }

      }
    }
  }
};

/*static*/ void BatchedCholeskyTask::omp_variant(TaskContext& context)
{
  openblas_set_num_threads(omp_get_max_threads());
  batched_cholesky_task_context_dispatch<VariantKind::OMP>(context);
}

}  // namespace cunumeric
