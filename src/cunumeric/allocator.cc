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

#include "cunumeric/allocator.h"

namespace cunumeric {

DeferredBufferAllocator::DeferredBufferAllocator(Legion::Memory::Kind kind) : target_kind(kind) {}

DeferredBufferAllocator::~DeferredBufferAllocator()
{
  for (auto& pair : buffers) { pair.second.destroy(); }
  buffers.clear();
}

char* DeferredBufferAllocator::allocate(size_t bytes)
{
  if (bytes == 0) return nullptr;

  // Use 16-byte alignment
  bytes = (bytes + 15) / 16 * 16;

  ByteBuffer buffer = legate::create_buffer<int8_t>(bytes, target_kind);

  void* ptr = buffer.ptr(0);
#ifdef DEBUG_CUNUMERIC
  assert(buffers.find(ptr) == buffers.end());
#endif
  buffers[ptr] = buffer;
  return (char*)ptr;
}

void DeferredBufferAllocator::deallocate(char* ptr, size_t n)
{
  ByteBuffer buffer;
  void* p     = ptr;
  auto finder = buffers.find(p);
#ifdef DEBUG_CUNUMERIC
  assert(finder != buffers.end() || removed.find(p) != removed.end());
#endif
  if (finder == buffers.end()) return;
  buffer = finder->second;
  buffers.erase(finder);
  buffer.destroy();
}

}  // namespace cunumeric