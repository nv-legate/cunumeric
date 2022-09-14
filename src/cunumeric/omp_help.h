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

#include <vector>

namespace cunumeric {

// Simple STL vector-based thread local storage for OpenMP threads to avoid false sharing
template <typename VAL>
struct ThreadLocalStorage {
 private:
  static constexpr size_t CACHE_LINE_SIZE = 64;

 public:
  ThreadLocalStorage(size_t num_threads)
    : storage_(CACHE_LINE_SIZE * num_threads), num_threads_(num_threads)
  {
    static_assert(sizeof(VAL) <= CACHE_LINE_SIZE, "ThreadLocalStorage element fits in cache line");
  }

  ~ThreadLocalStorage() {}

 public:
  VAL& operator[](size_t idx)
  {
    return *reinterpret_cast<VAL*>(storage_.data() + CACHE_LINE_SIZE * idx);
  }

 private:
  std::vector<int8_t> storage_;
  size_t num_threads_;
};

}  // namespace cunumeric
