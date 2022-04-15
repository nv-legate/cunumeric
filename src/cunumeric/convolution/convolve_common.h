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

#pragma once

#include "cunumeric/divmod.h"
#include "legion.h"

namespace cunumeric {

struct LoadComplexData {
  size_t buffervolume{0};

  __host__ inline bool update(size_t new_volume)
  {
    auto changed = buffervolume != new_volume;
    buffervolume = new_volume;
    return changed;
  }
};

struct ZeroPadLoadData {
  FastDivmodU64 pitches[3];
  size_t strides[3];
  size_t bounds[3];
  int32_t dim{0};
  size_t misalignment{0};

  template <int32_t DIM>
  __host__ inline bool update(const Legion::Point<DIM>& fftsize,
                              const size_t* new_strides,
                              const Legion::Point<DIM>& new_bounds,
                              size_t new_misalignment = 0)
  {
    auto changed = dim != DIM;
    changed      = true;
    dim          = DIM;
    size_t pitch = 1;
    for (int32_t d = DIM - 1; d >= 0; --d) {
      if (changed || (pitches[d].divisor != pitch)) {
        pitches[d] = FastDivmodU64(pitch);
        changed    = true;
      }
      pitch *= fftsize[d];
      if (changed || (strides[d] != new_strides[d])) {
        strides[d] = new_strides[d];
        changed    = true;
      }
      if (changed || (bounds[d] != new_bounds[d])) {
        bounds[d] = new_bounds[d];
        changed   = true;
      }
    }
    changed      = changed || misalignment != new_misalignment;
    misalignment = new_misalignment;
    return changed;
  }
};

template <typename T>
struct StoreOutputData : public ZeroPadLoadData {
  size_t offsets[3];
  T scale_factor;
  using ZeroPadLoadData::update;
  template <int32_t DIM>
  __host__ inline bool update(const Legion::Point<DIM>& fftsize,
                              const size_t* new_strides,
                              const Legion::Point<DIM>& new_offsets,
                              const Legion::Point<DIM>& new_bounds)
  {
    auto changed = update(fftsize, new_strides, new_bounds);
    for (int32_t d = DIM - 1; d >= 0; --d) {
      if (changed || offsets[d] != new_offsets[d]) {
        offsets[d] = new_offsets[d];
        changed    = true;
      }
    }
    if (changed) scale_factor = 1.0 / pitches[0].divisor;
    return changed;
  }
};

}  // namespace cunumeric
