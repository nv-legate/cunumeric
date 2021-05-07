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

#include "scalar.h"
#include "dispatch.h"

namespace legate {
namespace numpy {

UntypedScalar::UntypedScalar(const UntypedScalar &other) noexcept { copy(other); }

UntypedScalar::UntypedScalar(UntypedScalar &&other) noexcept
{
  move(std::forward<UntypedScalar>(other));
}

UntypedScalar::~UntypedScalar() { destroy(); }

UntypedScalar &UntypedScalar::operator=(const UntypedScalar &other) noexcept { copy(other); }

UntypedScalar &UntypedScalar::operator=(UntypedScalar &&other) noexcept
{
  move(std::forward<UntypedScalar>(other));
}

struct destroy_fn {
  template <LegateTypeCode CODE>
  void operator()(void *ptr)
  {
    delete static_cast<legate_type_of<CODE> *>(ptr);
  }
};

void UntypedScalar::destroy()
{
  if (nullptr != data_) {
    type_dispatch(code_, destroy_fn{}, data_);
    data_ = nullptr;
  }
}

struct copy_fn {
  template <LegateTypeCode CODE>
  void *operator()(const void *ptr)
  {
    using VAL = legate_type_of<CODE>;
    return new VAL(*static_cast<const VAL *>(ptr));
  }
};

void UntypedScalar::copy(const UntypedScalar &other)
{
  destroy();
  code_ = other.code_;
  data_ = type_dispatch(code_, copy_fn{}, other.data_);
}

void UntypedScalar::move(UntypedScalar &&other)
{
  destroy();
  code_       = other.code_;
  data_       = other.data_;
  other.code_ = LegateTypeCode::MAX_TYPE_NUMBER;
  other.data_ = nullptr;
}

size_t UntypedScalar::legion_buffer_size() const
{
  return sizeof(LegateTypeCode) + (nullptr != data_ ? elem_size() : 0);
}

void UntypedScalar::legion_serialize(void *buffer) const
{
  *static_cast<LegateTypeCode *>(buffer) = code_;
  if (nullptr != data_)
    memcpy(static_cast<int8_t *>(buffer) + sizeof(LegateTypeCode), data_, elem_size());
}

void UntypedScalar::legion_deserialize(const void *buffer)
{
  code_ = *static_cast<const LegateTypeCode *>(buffer);
  if (LegateTypeCode::MAX_TYPE_NUMBER != code_)
    data_ =
      type_dispatch(code_, copy_fn{}, static_cast<const int8_t *>(buffer) + sizeof(LegateTypeCode));
}

struct size_fn {
  template <LegateTypeCode CODE>
  size_t operator()()
  {
    return sizeof(legate_type_of<CODE>);
  }
};

size_t UntypedScalar::elem_size() const { return type_dispatch(code_, size_fn{}); }

void deserialize(Deserializer &ctx, UntypedScalar &scalar)
{
  FromFuture<UntypedScalar> fut_scalar;
  deserialize(ctx, fut_scalar);
  scalar = std::move(fut_scalar.value());
}

}  // namespace numpy
}  // namespace legate
