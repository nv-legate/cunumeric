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

namespace cunumeric {

template <VariantKind KIND, bool Dense = false>
struct BoolMaskPolicy {
};

template <>
struct BoolMaskPolicy<VariantKind::CPU, true> {
  template <class RECT, class AccessorRD, class Kernel>
  void operator()(const RECT& rect, const AccessorRD& mask, Kernel&& kernel)
  {
    const size_t volume = rect.volume();
    auto maskptr        = mask.ptr(rect);
    for (size_t idx = 0; idx < volume; ++idx) {
      if (maskptr[idx]) kernel(idx);
    }
  }
};

template <>
struct BoolMaskPolicy<VariantKind::CPU, false> {
  template <class RECT, class PITCHES, class AccessorRD, class Kernel>
  void operator()(const RECT& rect, const PITCHES& pitches, const AccessorRD& mask, Kernel&& kernel)
  {
    const size_t volume = rect.volume();
    for (size_t idx = 0; idx < volume; ++idx) {
      auto p = pitches.unflatten(idx, rect.lo);
      if (mask[p]) kernel(p);
    }
  }
};

}  // namespace cunumeric
