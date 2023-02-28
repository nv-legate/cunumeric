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

#include "cunumeric/cunumeric.h"
#include "cunumeric/bits/bits_util.h"

namespace cunumeric {

template <Bitorder BITORDER>
struct Unpack;

template <>
struct Unpack<Bitorder::BIG> {
  template <int32_t DIM>
  __CUDA_HD__ inline void operator()(legate::AccessorWO<uint8_t, DIM> out,
                                     legate::AccessorRO<uint8_t, DIM> in,
                                     legate::Point<DIM> p,
                                     uint32_t axis) const
  {
    int64_t out_hi = (p[axis] + 1) * 8 - 1;
    uint8_t val    = in[p];
    for (int32_t idx = 0; idx < 8; ++idx) {
      p[axis] = out_hi - idx;
      out[p]  = (val >> idx) & 0x01;
    }
  }
};

template <>
struct Unpack<Bitorder::LITTLE> {
  template <int32_t DIM>
  __CUDA_HD__ inline void operator()(legate::AccessorWO<uint8_t, DIM> out,
                                     legate::AccessorRO<uint8_t, DIM> in,
                                     legate::Point<DIM> p,
                                     uint32_t axis) const
  {
    int64_t out_lo = p[axis] * 8;
    uint8_t val    = in[p];
    for (int32_t idx = 0; idx < 8; ++idx) {
      p[axis] = out_lo + idx;
      out[p]  = (val >> idx) & 0x01;
    }
  }
};

class UnpackbitsTask : public CuNumericTask<UnpackbitsTask> {
 public:
  static const int TASK_ID = CUNUMERIC_UNPACKBITS;

 public:
  static void cpu_variant(legate::TaskContext& context);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& context);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

}  // namespace cunumeric
