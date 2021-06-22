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

#include "legate.h"
#include "legate_numpy_c.h"
#include "core.h"
#include "dispatch.h"
#include "mathtypes/complex.h"
#include <deque>

namespace legate {
namespace numpy {

using Array = Store;

template <LegateTypeCode CODE>
struct LegateTypeOf {
  using type = void;
};
template <>
struct LegateTypeOf<LegateTypeCode::BOOL_LT> {
  using type = bool;
};
template <>
struct LegateTypeOf<LegateTypeCode::INT8_LT> {
  using type = int8_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::INT16_LT> {
  using type = int16_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::INT32_LT> {
  using type = int32_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::INT64_LT> {
  using type = int64_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::UINT8_LT> {
  using type = uint8_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::UINT16_LT> {
  using type = uint16_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::UINT32_LT> {
  using type = uint32_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::UINT64_LT> {
  using type = uint64_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::HALF_LT> {
  using type = __half;
};
template <>
struct LegateTypeOf<LegateTypeCode::FLOAT_LT> {
  using type = float;
};
template <>
struct LegateTypeOf<LegateTypeCode::DOUBLE_LT> {
  using type = double;
};
template <>
struct LegateTypeOf<LegateTypeCode::COMPLEX64_LT> {
  using type = complex<float>;
};
template <>
struct LegateTypeOf<LegateTypeCode::COMPLEX128_LT> {
  using type = complex<double>;
};

template <LegateTypeCode CODE>
using legate_type_of = typename LegateTypeOf<CODE>::type;

template <LegateTypeCode CODE>
struct is_integral {
  static constexpr bool value = std::is_integral<legate_type_of<CODE>>::value;
};

template <LegateTypeCode CODE>
struct is_signed {
  static constexpr bool value = std::is_signed<legate_type_of<CODE>>::value;
};

template <LegateTypeCode CODE>
struct is_unsigned {
  static constexpr bool value = std::is_unsigned<legate_type_of<CODE>>::value;
};

template <LegateTypeCode CODE>
struct is_floating_point {
  static constexpr bool value = std::is_floating_point<legate_type_of<CODE>>::value;
};

template <typename T>
struct is_complex : std::false_type {
};

template <>
struct is_complex<complex<float>> : std::true_type {
};

template <>
struct is_complex<complex<double>> : std::true_type {
};

enum class VariantKind : int {
  CPU = 0,
  OMP = 1,
  GPU = 2,
};

static const char* TYPE_NAMES[] = {"bool",
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

class LegateNumPy {
 public:
  // Record variants for all our tasks
  static void record_variant(Legion::TaskID tid,
                             const char* task_name,
                             const Legion::CodeDescriptor& desc,
                             Legion::ExecutionConstraintSet& execution_constraints,
                             Legion::TaskLayoutConstraintSet& layout_constraints,
                             LegateVariant var,
                             Legion::Processor::Kind kind,
                             bool leaf,
                             bool inner,
                             bool idempotent,
                             size_t ret_size);

 public:
  struct PendingTaskVariant : public Legion::TaskVariantRegistrar {
   public:
    PendingTaskVariant(void)
      : Legion::TaskVariantRegistrar(), task_name(NULL), var(LEGATE_NO_VARIANT)
    {
    }
    PendingTaskVariant(Legion::TaskID tid,
                       bool global,
                       const char* var_name,
                       const char* t_name,
                       const Legion::CodeDescriptor& desc,
                       LegateVariant v,
                       size_t ret)
      : Legion::TaskVariantRegistrar(tid, global, var_name),
        task_name(t_name),
        descriptor(desc),
        var(v),
        ret_size(ret)
    {
    }

   public:
    const char* task_name;
    Legion::CodeDescriptor descriptor;
    LegateVariant var;
    size_t ret_size;
  };
  static std::deque<PendingTaskVariant>& get_pending_task_variants(void);
};

template <typename T>
class NumPyTask : public LegateTask<T> {
 public:
  // Record variants for all our tasks
  static void record_variant(Legion::TaskID tid,
                             const Legion::CodeDescriptor& desc,
                             Legion::ExecutionConstraintSet& execution_constraints,
                             Legion::TaskLayoutConstraintSet& layout_constraints,
                             LegateVariant var,
                             Legion::Processor::Kind kind,
                             bool leaf,
                             bool inner,
                             bool idempotent,
                             size_t ret_size)
  {
    // For this just turn around and call this on the base LegateNumPy
    // type so it will deduplicate across all task kinds
    LegateNumPy::record_variant(tid,
                                NumPyTask<T>::task_name(),
                                desc,
                                execution_constraints,
                                layout_constraints,
                                var,
                                kind,
                                leaf,
                                inner,
                                idempotent,
                                ret_size);
  }
};

}  // namespace numpy
}  // namespace legate
