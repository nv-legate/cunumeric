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

#pragma once

#include "numpy.h"
#include "core.h"

namespace legate {
namespace numpy {

template <typename T>
struct Span {
 public:
  Span(T *data, size_t size) : data_(data), size_(size) {}

 public:
  decltype(auto) operator[](size_t pos)
  {
    assert(pos < size_);
    return data_[pos];
  }

 public:
  decltype(auto) subspan(size_t off)
  {
    assert(off <= size_);
    return Span(data_ + off, size_ - off);
  }

 private:
  T *data_;
  size_t size_;
};

class Deserializer {
 public:
  Deserializer(const Legion::Task *task, const std::vector<Legion::PhysicalRegion> &regions);

 public:
  friend void deserialize(Deserializer &ctx, __half &value);
  friend void deserialize(Deserializer &ctx, float &value);
  friend void deserialize(Deserializer &ctx, double &value);
  friend void deserialize(Deserializer &ctx, std::uint64_t &value);
  friend void deserialize(Deserializer &ctx, std::uint32_t &value);
  friend void deserialize(Deserializer &ctx, std::uint16_t &value);
  friend void deserialize(Deserializer &ctx, std::uint8_t &value);
  friend void deserialize(Deserializer &ctx, std::int64_t &value);
  friend void deserialize(Deserializer &ctx, std::int32_t &value);
  friend void deserialize(Deserializer &ctx, std::int16_t &value);
  friend void deserialize(Deserializer &ctx, std::int8_t &value);
  friend void deserialize(Deserializer &ctx, std::string &value);

 public:
  friend void deserialize(Deserializer &ctx, LegateTypeCode &code);

 public:
  friend void deserialize(Deserializer &ctx, Shape &value);
  friend void deserialize(Deserializer &ctx, Transform &value);
  friend void deserialize(Deserializer &ctx, RegionField &value);

 private:
  const Legion::Task *task_;
  Span<const Legion::PhysicalRegion> regions_;
  Span<const Legion::Future> futures_;
  LegateDeserializer deserializer_;
  std::vector<Legion::OutputRegion> outputs_;
};

}  // namespace numpy
}  // namespace legate
