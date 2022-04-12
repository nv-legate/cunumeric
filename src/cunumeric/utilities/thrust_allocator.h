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

#include "legate.h"

namespace cunumeric {

class ThrustAllocator : public ScopedAllocator {
 public:
  using value_type = char;

  ThrustAllocator(Legion::Memory::Kind kind) : ScopedAllocator(kind) {}

  char* allocate(size_t num_bytes) { return (char*)ScopedAllocator::allocate(num_bytes); }

  void deallocate(char* ptr, size_t n) { ScopedAllocator::deallocate(ptr); }
};

}  // namespace cunumeric
