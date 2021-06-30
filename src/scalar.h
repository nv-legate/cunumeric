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
#include "arg.h"
#include "deserializer.h"

namespace legate {
namespace numpy {

class UntypedScalar {
 public:
  UntypedScalar() = default;
  UntypedScalar(const UntypedScalar &other) noexcept;
  UntypedScalar(UntypedScalar &&other) noexcept;
  ~UntypedScalar();

 public:
  UntypedScalar &operator=(const UntypedScalar &other) noexcept;
  UntypedScalar &operator=(UntypedScalar &&other) noexcept;

 public:
  template <typename T>
  UntypedScalar(const T &value) : is_argval_(0), code_(legate_type_code_of<T>), data_(new T(value))
  {
  }
  template <typename T>
  UntypedScalar(const Argval<T> &value)
    : is_argval_(1), code_(legate_type_code_of<T>), data_(new Argval<T>(value))
  {
  }

 private:
  void destroy();
  void copy(const UntypedScalar &other);
  void move(UntypedScalar &&other);

 public:
  size_t legion_buffer_size() const;
  void legion_serialize(void *buffer) const;
  void legion_deserialize(const void *buffer);

 public:
  auto code() const { return code_; }
  size_t elem_size() const;
  bool is_argval() const { return is_argval_ != 0; }

 public:
  template <typename T>
  const T &value() const
  {
    return *static_cast<const T *>(data_);
  }
  template <typename T = void>
  const T *ptr() const
  {
    return static_cast<const T *>(data_);
  }

  template <typename T = void>
  T *ptr()
  {
    return static_cast<T *>(data_);
  }

  std::string to_string() const;

 private:
  int32_t is_argval_{0};
  LegateTypeCode code_{LegateTypeCode::MAX_TYPE_NUMBER};
  void *data_{nullptr};
};

void deserialize(Deserializer &ctx, UntypedScalar &scalar);

}  // namespace numpy

template <>
struct ReturnSize<numpy::UntypedScalar> {
  static constexpr int32_t value = sizeof(complex<double>) + 8;
};

}  // namespace legate
