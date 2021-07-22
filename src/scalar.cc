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

#include <sstream>

#include "scalar.h"

namespace legate {
namespace numpy {

static const char *TYPE_NAMES[] = {"bool",
                                   "int8_t",
                                   "int16_t",
                                   "int32_t",
                                   "int64_t",
                                   "uint8_t",
                                   "uint16_t",
                                   "uint32_t",
                                   "uint64_t",
                                   "__half",
                                   "float",
                                   "double",
                                   "complex<float>",
                                   "complex<double>"};

UntypedScalar::UntypedScalar(const UntypedScalar &other) noexcept { copy(other); }

UntypedScalar::UntypedScalar(UntypedScalar &&other) noexcept
{
  move(std::forward<UntypedScalar>(other));
}

UntypedScalar::~UntypedScalar() { destroy(); }

UntypedScalar &UntypedScalar::operator=(const UntypedScalar &other) noexcept
{
  copy(other);
  return *this;
}

UntypedScalar &UntypedScalar::operator=(UntypedScalar &&other) noexcept
{
  move(std::forward<UntypedScalar>(other));
  return *this;
}

struct destroy_fn {
  template <LegateTypeCode CODE>
  void operator()(bool is_argval, void *ptr)
  {
    using VAL = legate_type_of<CODE>;
    if (is_argval)
      delete static_cast<Argval<VAL> *>(ptr);
    else
      delete static_cast<VAL *>(ptr);
  }
};

void UntypedScalar::destroy()
{
  if (nullptr != data_) {
    type_dispatch(code_, destroy_fn{}, is_argval_, data_);
    data_ = nullptr;
  }
}

struct copy_fn {
  template <LegateTypeCode CODE>
  void *operator()(bool is_argval, const void *ptr)
  {
    using VAL = legate_type_of<CODE>;
    if (is_argval)
      return new Argval<VAL>(*static_cast<const Argval<VAL> *>(ptr));
    else
      return new VAL(*static_cast<const VAL *>(ptr));
  }
};

void UntypedScalar::copy(const UntypedScalar &other)
{
  destroy();
  if (nullptr == other.data_) return;
  is_argval_ = other.is_argval_;
  code_      = other.code_;
  data_      = type_dispatch(code_, copy_fn{}, other.is_argval_, other.data_);
}

void UntypedScalar::move(UntypedScalar &&other)
{
  destroy();
  is_argval_ = other.is_argval_;
  code_      = other.code_;
  data_      = other.data_;
  assert(nullptr != data_ || code_ == LegateTypeCode::MAX_TYPE_NUMBER);
  other.is_argval_ = false;
  other.code_      = LegateTypeCode::MAX_TYPE_NUMBER;
  other.data_      = nullptr;
}

size_t UntypedScalar::legion_buffer_size() const
{
  return sizeof(uint64_t) + (nullptr != data_ ? elem_size() : 0);
}

void UntypedScalar::legion_serialize(void *buffer) const
{
  *static_cast<int32_t *>(buffer)        = is_argval_;
  buffer                                 = static_cast<int8_t *>(buffer) + sizeof(int32_t);
  *static_cast<LegateTypeCode *>(buffer) = code_;
  if (nullptr != data_)
    memcpy(static_cast<int8_t *>(buffer) + sizeof(LegateTypeCode), data_, elem_size());
}

void UntypedScalar::legion_deserialize(const void *buffer)
{
  is_argval_ = *static_cast<const int32_t *>(buffer);
  buffer     = static_cast<const int8_t *>(buffer) + sizeof(int32_t);
  code_      = *static_cast<const LegateTypeCode *>(buffer);
  if (LegateTypeCode::MAX_TYPE_NUMBER != code_)
    data_ = type_dispatch(
      code_, copy_fn{}, is_argval_, static_cast<const int8_t *>(buffer) + sizeof(LegateTypeCode));
}

struct size_fn {
  template <LegateTypeCode CODE>
  size_t operator()(bool is_argval)
  {
    if (is_argval)
      return sizeof(Argval<legate_type_of<CODE>>);
    else
      return sizeof(legate_type_of<CODE>);
  }
};

size_t UntypedScalar::elem_size() const { return type_dispatch(code_, size_fn{}, is_argval_); }

struct to_string_fn {
  template <LegateTypeCode CODE>
  std::string operator()(bool is_argval, const void *data)
  {
    using VAL = legate_type_of<CODE>;
    std::stringstream ss;
    ss << TYPE_NAMES[CODE] << "(";
    if (is_argval) {
      auto val = static_cast<const Argval<VAL> *>(data);
      ss << "<" << val->arg << ", " << val->arg_value << ">)";
    } else
      ss << *static_cast<const VAL *>(data) << ")";
    return ss.str();
  }
};

std::string UntypedScalar::to_string() const
{
  if (nullptr == data_)
    return "invalid";
  else
    return type_dispatch(code_, to_string_fn{}, is_argval_, data_);
}

}  // namespace numpy
}  // namespace legate
