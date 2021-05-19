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
#include "mathtypes/complex.h"
#include <deque>

#ifndef NUMPY_RADIX
#define NUMPY_RADIX 4
#endif

#ifndef MAX_REDUCTION_RADIX
#define MAX_REDUCTION_RADIX 8
#endif

// Some help for indexing for broadcasting
#define COORD_MASK ((Legion::coord_t)ULONG_LONG_MAX)

namespace legate {
namespace numpy {

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
                             bool ret_type);

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
                       bool ret)
      : Legion::TaskVariantRegistrar(tid, global, var_name),
        task_name(t_name),
        descriptor(desc),
        var(v),
        ret_type(ret)
    {
    }

   public:
    const char* task_name;
    Legion::CodeDescriptor descriptor;
    LegateVariant var;
    bool ret_type;
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
                             bool ret_type)
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
                                ret_type);
  }
};

}  // namespace numpy
}  // namespace legate

#define NUMPY_TYPE_OFFSET (MAX_TYPE_NUMBER * NUMPY_MAX_VARIANTS)

#define INSTANTIATE_ALL_TASKS(type, base_id)                                            \
  template <>                                                                           \
  const int type<float>::TASK_ID = base_id + FLOAT_LT* NUMPY_MAX_VARIANTS;              \
  template class type<float>;                                                           \
  template <>                                                                           \
  const int type<double>::TASK_ID = base_id + DOUBLE_LT* NUMPY_MAX_VARIANTS;            \
  template class type<double>;                                                          \
  template <>                                                                           \
  const int type<int16_t>::TASK_ID = base_id + INT16_LT* NUMPY_MAX_VARIANTS;            \
  template class type<int16_t>;                                                         \
  template <>                                                                           \
  const int type<int32_t>::TASK_ID = base_id + INT32_LT* NUMPY_MAX_VARIANTS;            \
  template class type<int32_t>;                                                         \
  template <>                                                                           \
  const int type<int64_t>::TASK_ID = base_id + INT64_LT* NUMPY_MAX_VARIANTS;            \
  template class type<int64_t>;                                                         \
  template <>                                                                           \
  const int type<uint16_t>::TASK_ID = base_id + UINT16_LT* NUMPY_MAX_VARIANTS;          \
  template class type<uint16_t>;                                                        \
  template <>                                                                           \
  const int type<uint32_t>::TASK_ID = base_id + UINT32_LT* NUMPY_MAX_VARIANTS;          \
  template class type<uint32_t>;                                                        \
  template <>                                                                           \
  const int type<uint64_t>::TASK_ID = base_id + UINT64_LT* NUMPY_MAX_VARIANTS;          \
  template class type<uint64_t>;                                                        \
  template <>                                                                           \
  const int type<bool>::TASK_ID = base_id + BOOL_LT* NUMPY_MAX_VARIANTS;                \
  template class type<bool>;                                                            \
  template <>                                                                           \
  const int type<__half>::TASK_ID = base_id + HALF_LT* NUMPY_MAX_VARIANTS;              \
  template class type<__half>;                                                          \
  template <>                                                                           \
  const int type<complex<float>>::TASK_ID = base_id + COMPLEX64_LT* NUMPY_MAX_VARIANTS; \
  template class type<complex<float>>;                                                  \
  // template<>
  // const int type<complex<double>>::TASK_ID = base_id + COMPLEX128_LT* NUMPY_MAX_VARIANTS; \
  // template class type<complex<double>>;

#define INSTANTIATE_NONCOMPLEX_TASKS(type, base_id)                            \
  template <>                                                                  \
  const int type<float>::TASK_ID = base_id + FLOAT_LT* NUMPY_MAX_VARIANTS;     \
  template class type<float>;                                                  \
  template <>                                                                  \
  const int type<double>::TASK_ID = base_id + DOUBLE_LT* NUMPY_MAX_VARIANTS;   \
  template class type<double>;                                                 \
  template <>                                                                  \
  const int type<int16_t>::TASK_ID = base_id + INT16_LT* NUMPY_MAX_VARIANTS;   \
  template class type<int16_t>;                                                \
  template <>                                                                  \
  const int type<int32_t>::TASK_ID = base_id + INT32_LT* NUMPY_MAX_VARIANTS;   \
  template class type<int32_t>;                                                \
  template <>                                                                  \
  const int type<int64_t>::TASK_ID = base_id + INT64_LT* NUMPY_MAX_VARIANTS;   \
  template class type<int64_t>;                                                \
  template <>                                                                  \
  const int type<uint16_t>::TASK_ID = base_id + UINT16_LT* NUMPY_MAX_VARIANTS; \
  template class type<uint16_t>;                                               \
  template <>                                                                  \
  const int type<uint32_t>::TASK_ID = base_id + UINT32_LT* NUMPY_MAX_VARIANTS; \
  template class type<uint32_t>;                                               \
  template <>                                                                  \
  const int type<uint64_t>::TASK_ID = base_id + UINT64_LT* NUMPY_MAX_VARIANTS; \
  template class type<uint64_t>;                                               \
  template <>                                                                  \
  const int type<bool>::TASK_ID = base_id + BOOL_LT* NUMPY_MAX_VARIANTS;       \
  template class type<bool>;                                                   \
  template <>                                                                  \
  const int type<__half>::TASK_ID = base_id + HALF_LT* NUMPY_MAX_VARIANTS;     \
  template class type<__half>;

#define INSTANTIATE_INT_TASKS(type, base_id)                                 \
  template <>                                                                \
  const int type<int16_t>::TASK_ID = base_id + INT16_LT* NUMPY_MAX_VARIANTS; \
  template class type<int16_t>;                                              \
  template <>                                                                \
  const int type<int32_t>::TASK_ID = base_id + INT32_LT* NUMPY_MAX_VARIANTS; \
  template class type<int32_t>;                                              \
  template <>                                                                \
  const int type<int64_t>::TASK_ID = base_id + INT64_LT* NUMPY_MAX_VARIANTS; \
  template class type<int64_t>;

#define INSTANTIATE_UINT_TASKS(type, base_id)                                  \
  template <>                                                                  \
  const int type<uint16_t>::TASK_ID = base_id + UINT16_LT* NUMPY_MAX_VARIANTS; \
  template class type<uint16_t>;                                               \
  template <>                                                                  \
  const int type<uint32_t>::TASK_ID = base_id + UINT32_LT* NUMPY_MAX_VARIANTS; \
  template class type<uint32_t>;                                               \
  template <>                                                                  \
  const int type<uint64_t>::TASK_ID = base_id + UINT64_LT* NUMPY_MAX_VARIANTS; \
  template class type<uint64_t>;

#define INSTANTIATE_REAL_TASKS(type, base_id)                                \
  template <>                                                                \
  const int type<float>::TASK_ID = base_id + FLOAT_LT* NUMPY_MAX_VARIANTS;   \
  template class type<float>;                                                \
  template <>                                                                \
  const int type<double>::TASK_ID = base_id + DOUBLE_LT* NUMPY_MAX_VARIANTS; \
  template class type<double>;                                               \
  template <>                                                                \
  const int type<__half>::TASK_ID = base_id + HALF_LT* NUMPY_MAX_VARIANTS;   \
  template class type<__half>;

#define INSTANTIATE_COMPLEX_TASKS(type, base_id)                                        \
  template <>                                                                           \
  const int type<complex<float>>::TASK_ID = base_id + COMPLEX64_LT* NUMPY_MAX_VARIANTS; \
  template class type<complex<float>>;                                                  \
  // template<>
  // const int type<complex<double>>::TASK_ID = base_id + COMPLEX128_LT* NUMPY_MAX_VARIANTS; \
  // template class type<complex<double>>;

#define INSTANTIATE_TASK_VARIANT(type, variant)                          \
  template void type<float>::variant(                                    \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void type<double>::variant(                                   \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void type<int16_t>::variant(                                  \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void type<int32_t>::variant(                                  \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void type<int64_t>::variant(                                  \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void type<uint16_t>::variant(                                 \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void type<uint32_t>::variant(                                 \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void type<uint64_t>::variant(                                 \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void type<bool>::variant(                                     \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void type<__half>::variant(                                   \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void type<complex<float>>::variant(                           \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
   // template void type<complex<double>>::variant(const Task*, const std::vector<PhysicalRegion>&,
   // Context, Runtime*);

#define INSTANTIATE_INT_VARIANT(type, variant)                           \
  template void type<int16_t>::variant(                                  \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void type<int32_t>::variant(                                  \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void type<int64_t>::variant(                                  \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);

#define INSTANTIATE_UINT_VARIANT(type, variant)                          \
  template void type<uint16_t>::variant(                                 \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void type<uint32_t>::variant(                                 \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void type<uint64_t>::variant(                                 \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);

#define INSTANTIATE_REAL_VARIANT(type, variant)                          \
  template void type<float>::variant(                                    \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void type<double>::variant(                                   \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void type<__half>::variant(                                   \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);

#define INSTANTIATE_COMPLEX_VARIANT(type, variant)                       \
  template void type<complex<float>>::variant(                           \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template void type<complex<double>>::variant(                          \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);

#define INSTANTIATE_NONFLOAT_TASKS(type, base_id)                              \
  const int type<__half>::TASK_ID = base_id + HALF_LT * NUMPY_MAX_VARIANTS;    \
  const int type<float>::TASK_ID  = base_id + FLOAT_LT * NUMPY_MAX_VARIANTS;   \
  template <>                                                                  \
  const int type<double>::TASK_ID = base_id + DOUBLE_LT* NUMPY_MAX_VARIANTS;   \
  template class type<double>;                                                 \
  template <>                                                                  \
  const int type<int16_t>::TASK_ID = base_id + INT16_LT* NUMPY_MAX_VARIANTS;   \
  template class type<int16_t>;                                                \
  template <>                                                                  \
  const int type<int32_t>::TASK_ID = base_id + INT32_LT* NUMPY_MAX_VARIANTS;   \
  template class type<int32_t>;                                                \
  template <>                                                                  \
  const int type<int64_t>::TASK_ID = base_id + INT64_LT* NUMPY_MAX_VARIANTS;   \
  template class type<int64_t>;                                                \
  template <>                                                                  \
  const int type<uint16_t>::TASK_ID = base_id + UINT16_LT* NUMPY_MAX_VARIANTS; \
  template class type<uint16_t>;                                               \
  template <>                                                                  \
  const int type<uint32_t>::TASK_ID = base_id + UINT32_LT* NUMPY_MAX_VARIANTS; \
  template class type<uint32_t>;                                               \
  template <>                                                                  \
  const int type<uint64_t>::TASK_ID = base_id + UINT64_LT* NUMPY_MAX_VARIANTS; \
  template class type<uint64_t>;                                               \
  template <>                                                                  \
  const int type<bool>::TASK_ID = base_id + BOOL_LT* NUMPY_MAX_VARIANTS;       \
  template class type<bool>;

#define INSTANTIATE_DEFERRED_REDUCTION_TASK_VARIANT(type, redop, variant)          \
  template DeferredReduction<redop<float>> type<float>::variant(                   \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);           \
  template DeferredReduction<redop<double>> type<double>::variant(                 \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);           \
  template DeferredReduction<redop<int16_t>> type<int16_t>::variant(               \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);           \
  template DeferredReduction<redop<int32_t>> type<int32_t>::variant(               \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);           \
  template DeferredReduction<redop<int64_t>> type<int64_t>::variant(               \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);           \
  template DeferredReduction<redop<uint16_t>> type<uint16_t>::variant(             \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);           \
  template DeferredReduction<redop<uint32_t>> type<uint32_t>::variant(             \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);           \
  template DeferredReduction<redop<uint64_t>> type<uint64_t>::variant(             \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);           \
  template DeferredReduction<redop<bool>> type<bool>::variant(                     \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);           \
  template DeferredReduction<redop<__half>> type<__half>::variant(                 \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);           \
  template DeferredReduction<redop<complex<float>>> type<complex<float>>::variant( \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
// template DeferredReduction<redop<complex<double>>
// type<complex<double>>::variant(const Task*, const std::vector<PhysicalRegion>&, Context,
// Runtime*);

#define INSTANTIATE_DEFERRED_REDUCTION_ARG_RETURN_TASK_VARIANT(type, redop, variant) \
  template DeferredReduction<redop<uint64_t>> type<float>::variant(                  \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);             \
  template DeferredReduction<redop<uint64_t>> type<double>::variant(                 \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);             \
  template DeferredReduction<redop<uint64_t>> type<int16_t>::variant(                \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);             \
  template DeferredReduction<redop<uint64_t>> type<int32_t>::variant(                \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);             \
  template DeferredReduction<redop<uint64_t>> type<int64_t>::variant(                \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);             \
  template DeferredReduction<redop<uint64_t>> type<uint16_t>::variant(               \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);             \
  template DeferredReduction<redop<uint64_t>> type<uint32_t>::variant(               \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);             \
  template DeferredReduction<redop<uint64_t>> type<uint64_t>::variant(               \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);             \
  template DeferredReduction<redop<uint64_t>> type<bool>::variant(                   \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);             \
  template DeferredReduction<redop<uint64_t>> type<__half>::variant(                 \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);             \
  template DeferredReduction<redop<uint64_t>> type<complex<float>>::variant(         \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
// template DeferredReduction<redop<uint64_t>>
// type<complex<double>>::variant(const Task*, const std::vector<PhysicalRegion>&, Context,
// Runtime*);

#define INSTANTIATE_DEFERRED_VALUE_TASK_VARIANTS(type, variant)          \
  template DeferredValue<float> type<float>::variant(                    \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template DeferredValue<double> type<double>::variant(                  \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template DeferredValue<int16_t> type<int16_t>::variant(                \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template DeferredValue<int32_t> type<int32_t>::variant(                \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template DeferredValue<int64_t> type<int64_t>::variant(                \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template DeferredValue<uint16_t> type<uint16_t>::variant(              \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template DeferredValue<uint32_t> type<uint32_t>::variant(              \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template DeferredValue<uint64_t> type<uint64_t>::variant(              \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template DeferredValue<bool> type<bool>::variant(                      \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template DeferredValue<__half> type<__half>::variant(                  \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*); \
  template DeferredValue<complex<float>> type<complex<float>>::variant(  \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
// template DeferredValue<complex<double>>
// type<complex<double>>::variant(const Task*, const std::vector<PhysicalRegion>&, Context,
// Runtime*);

#define INSTANTIATE_DEFERRED_VALUE_TASK_VARIANT(type, return_type, variant) \
  template DeferredValue<return_type> type<float>::variant(                 \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);    \
  template DeferredValue<return_type> type<double>::variant(                \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);    \
  template DeferredValue<return_type> type<int16_t>::variant(               \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);    \
  template DeferredValue<return_type> type<int32_t>::variant(               \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);    \
  template DeferredValue<return_type> type<int64_t>::variant(               \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);    \
  template DeferredValue<return_type> type<uint16_t>::variant(              \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);    \
  template DeferredValue<return_type> type<uint32_t>::variant(              \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);    \
  template DeferredValue<return_type> type<uint64_t>::variant(              \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);    \
  template DeferredValue<return_type> type<bool>::variant(                  \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);    \
  template DeferredValue<return_type> type<__half>::variant(                \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);    \
  template DeferredValue<return_type> type<complex<float>>::variant(        \
    const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);
// template DeferredValue<return_type>
// type<complex<double>>::variant(const Task*, const std::vector<PhysicalRegion>&, Context,
// Runtime*);

#define REGISTER_ALL_TASKS(type)               \
  {                                            \
    type<float>::register_variants();          \
    type<double>::register_variants();         \
    type<int16_t>::register_variants();        \
    type<int32_t>::register_variants();        \
    type<int64_t>::register_variants();        \
    type<uint16_t>::register_variants();       \
    type<uint32_t>::register_variants();       \
    type<uint64_t>::register_variants();       \
    type<bool>::register_variants();           \
    type<__half>::register_variants();         \
    type<complex<float>>::register_variants(); \
  }

#define REGISTER_NONCOMPLEX_TASKS(type)  \
  {                                      \
    type<float>::register_variants();    \
    type<double>::register_variants();   \
    type<int16_t>::register_variants();  \
    type<int32_t>::register_variants();  \
    type<int64_t>::register_variants();  \
    type<uint16_t>::register_variants(); \
    type<uint32_t>::register_variants(); \
    type<uint64_t>::register_variants(); \
    type<bool>::register_variants();     \
    type<__half>::register_variants();   \
  }

#define REGISTER_INT_TASKS(type)        \
  {                                     \
    type<int16_t>::register_variants(); \
    type<int32_t>::register_variants(); \
    type<int64_t>::register_variants(); \
  }

#define REGISTER_UINT_TASKS(type)        \
  {                                      \
    type<uint16_t>::register_variants(); \
    type<uint32_t>::register_variants(); \
    type<uint64_t>::register_variants(); \
  }

#define REGISTER_REAL_TASKS(type)      \
  {                                    \
    type<float>::register_variants();  \
    type<double>::register_variants(); \
    type<__half>::register_variants(); \
  }

#define REGISTER_ALL_TASKS_WITH_RETURN(type)                                               \
  {                                                                                        \
    type<float>::register_variants_with_return<float, float>();                            \
    type<double>::register_variants_with_return<double, double>();                         \
    type<int16_t>::register_variants_with_return<int16_t, int16_t>();                      \
    type<int32_t>::register_variants_with_return<int32_t, int32_t>();                      \
    type<int64_t>::register_variants_with_return<int64_t, int64_t>();                      \
    type<uint16_t>::register_variants_with_return<uint16_t, uint16_t>();                   \
    type<uint32_t>::register_variants_with_return<uint32_t, uint32_t>();                   \
    type<uint64_t>::register_variants_with_return<uint64_t, uint64_t>();                   \
    type<bool>::register_variants_with_return<bool, bool>();                               \
    type<__half>::register_variants_with_return<__half, __half>();                         \
    type<complex<float>>::register_variants_with_return<complex<float>, complex<float>>(); \
  }
   // type<complex<double>>::register_variants_with_return<complex<double>, complex<double>>(); \

#define REGISTER_NONCOMPLEX_TASKS_WITH_RETURN(type)                      \
  {                                                                      \
    type<float>::register_variants_with_return<float, float>();          \
    type<double>::register_variants_with_return<double, double>();       \
    type<int16_t>::register_variants_with_return<int16_t, int16_t>();    \
    type<int32_t>::register_variants_with_return<int32_t, int32_t>();    \
    type<int64_t>::register_variants_with_return<int64_t, int64_t>();    \
    type<uint16_t>::register_variants_with_return<uint16_t, uint16_t>(); \
    type<uint32_t>::register_variants_with_return<uint32_t, uint32_t>(); \
    type<uint64_t>::register_variants_with_return<uint64_t, uint64_t>(); \
    type<bool>::register_variants_with_return<bool, bool>();             \
    type<__half>::register_variants_with_return<__half, __half>();       \
  }

#define REGISTER_ALL_TASKS_WITH_BOOL_RETURN(type)                                     \
  {                                                                                   \
    type<float>::register_variants_with_return<bool, DeferredValue<bool>>();          \
    type<double>::register_variants_with_return<bool, DeferredValue<bool>>();         \
    type<int16_t>::register_variants_with_return<bool, DeferredValue<bool>>();        \
    type<int32_t>::register_variants_with_return<bool, DeferredValue<bool>>();        \
    type<int64_t>::register_variants_with_return<bool, DeferredValue<bool>>();        \
    type<uint16_t>::register_variants_with_return<bool, DeferredValue<bool>>();       \
    type<uint32_t>::register_variants_with_return<bool, DeferredValue<bool>>();       \
    type<uint64_t>::register_variants_with_return<bool, DeferredValue<bool>>();       \
    type<bool>::register_variants_with_return<bool, DeferredValue<bool>>();           \
    type<__half>::register_variants_with_return<bool, DeferredValue<bool>>();         \
    type<complex<float>>::register_variants_with_return<bool, DeferredValue<bool>>(); \
  }
   // type<complex<double>>::register_variants_with_return<bool, DeferredValue<bool>>();   \

#define REGISTER_NONCOMPLEX_TASKS_WITH_BOOL_RETURN(type)                        \
  {                                                                             \
    type<float>::register_variants_with_return<bool, DeferredValue<bool>>();    \
    type<double>::register_variants_with_return<bool, DeferredValue<bool>>();   \
    type<int16_t>::register_variants_with_return<bool, DeferredValue<bool>>();  \
    type<int32_t>::register_variants_with_return<bool, DeferredValue<bool>>();  \
    type<int64_t>::register_variants_with_return<bool, DeferredValue<bool>>();  \
    type<uint16_t>::register_variants_with_return<bool, DeferredValue<bool>>(); \
    type<uint32_t>::register_variants_with_return<bool, DeferredValue<bool>>(); \
    type<uint64_t>::register_variants_with_return<bool, DeferredValue<bool>>(); \
    type<bool>::register_variants_with_return<bool, DeferredValue<bool>>();     \
    type<__half>::register_variants_with_return<bool, DeferredValue<bool>>();   \
  }

#define REGISTER_REAL_TASKS_WITH_BOOL_RETURN(type)                            \
  {                                                                           \
    type<float>::register_variants_with_return<bool, DeferredValue<bool>>();  \
    type<double>::register_variants_with_return<bool, DeferredValue<bool>>(); \
    type<__half>::register_variants_with_return<bool, DeferredValue<bool>>(); \
  }

#define REGISTER_ALL_TASKS_WITH_FLOAT_RETURN(type)                                        \
  {                                                                                       \
    type<float>::register_variants_with_return<float, DeferredValue<float>>();            \
    type<double>::register_variants_with_return<double, DeferredValue<double>>();         \
    type<int16_t>::register_variants_with_return<double, DeferredValue<double>>();        \
    type<int32_t>::register_variants_with_return<double, DeferredValue<double>>();        \
    type<int64_t>::register_variants_with_return<double, DeferredValue<double>>();        \
    type<uint16_t>::register_variants_with_return<double, DeferredValue<double>>();       \
    type<uint32_t>::register_variants_with_return<double, DeferredValue<double>>();       \
    type<uint64_t>::register_variants_with_return<double, DeferredValue<double>>();       \
    type<bool>::register_variants_with_return<double, DeferredValue<double>>();           \
    type<__half>::register_variants_with_return<__half, DeferredValue<__half>>();         \
    type<complex<float>>::register_variants_with_return<complex<float>,                   \
                                                        DeferredValue<complex<float>>>(); \
  }
   // type<complex<double>>::register_variants_with_return<complex<double>,
   // DeferredValue<complex<double>>>();   \

#define REGISTER_ALL_TASKS_WITH_ARG_RETURN(type)                                            \
  {                                                                                         \
    type<float>::register_variants_with_return<int64_t, DeferredValue<int64_t>>();          \
    type<double>::register_variants_with_return<int64_t, DeferredValue<int64_t>>();         \
    type<int16_t>::register_variants_with_return<int64_t, DeferredValue<int64_t>>();        \
    type<int32_t>::register_variants_with_return<int64_t, DeferredValue<int64_t>>();        \
    type<int64_t>::register_variants_with_return<int64_t, DeferredValue<int64_t>>();        \
    type<uint16_t>::register_variants_with_return<int64_t, DeferredValue<int64_t>>();       \
    type<uint32_t>::register_variants_with_return<int64_t, DeferredValue<int64_t>>();       \
    type<uint64_t>::register_variants_with_return<int64_t, DeferredValue<int64_t>>();       \
    type<bool>::register_variants_with_return<int64_t, DeferredValue<int64_t>>();           \
    type<__half>::register_variants_with_return<int64_t, DeferredValue<int64_t>>();         \
    type<complex<float>>::register_variants_with_return<int64_t, DeferredValue<int64_t>>(); \
  }
   // type<complex<double>>::register_variants_with_return<int64_t, DeferredValue<int64_t>>();   \

#define REGISTER_ALL_TASKS_WITH_VALUE_RETURN(type)                                        \
  {                                                                                       \
    type<float>::register_variants_with_return<float, DeferredValue<float>>();            \
    type<double>::register_variants_with_return<double, DeferredValue<double>>();         \
    type<int16_t>::register_variants_with_return<int16_t, DeferredValue<int16_t>>();      \
    type<int32_t>::register_variants_with_return<int32_t, DeferredValue<int32_t>>();      \
    type<int64_t>::register_variants_with_return<int64_t, DeferredValue<int64_t>>();      \
    type<uint16_t>::register_variants_with_return<uint16_t, DeferredValue<uint16_t>>();   \
    type<uint32_t>::register_variants_with_return<uint32_t, DeferredValue<uint32_t>>();   \
    type<uint64_t>::register_variants_with_return<uint64_t, DeferredValue<uint64_t>>();   \
    type<bool>::register_variants_with_return<bool, DeferredValue<bool>>();               \
    type<__half>::register_variants_with_return<__half, DeferredValue<__half>>();         \
    type<complex<float>>::register_variants_with_return<complex<float>,                   \
                                                        DeferredValue<complex<float>>>(); \
  }
   // type<complex<double>>::register_variants_with_return<complex<double>,
   // DeferredValue<complex<double>>>();       \

#define REGISTER_ALL_TASKS_WITH_REDUCTION_RETURN(type, redop)                                      \
  {                                                                                                \
    type<float>::register_variants_with_return<float, DeferredReduction<redop<float>>>();          \
    type<double>::register_variants_with_return<double, DeferredReduction<redop<double>>>();       \
    type<int16_t>::register_variants_with_return<int16_t, DeferredReduction<redop<int16_t>>>();    \
    type<int32_t>::register_variants_with_return<int32_t, DeferredReduction<redop<int32_t>>>();    \
    type<int64_t>::register_variants_with_return<int64_t, DeferredReduction<redop<int64_t>>>();    \
    type<uint16_t>::register_variants_with_return<uint16_t, DeferredReduction<redop<uint16_t>>>(); \
    type<uint32_t>::register_variants_with_return<uint32_t, DeferredReduction<redop<uint32_t>>>(); \
    type<uint64_t>::register_variants_with_return<uint64_t, DeferredReduction<redop<uint64_t>>>(); \
    type<bool>::register_variants_with_return<bool, DeferredReduction<redop<bool>>>();             \
    type<__half>::register_variants_with_return<__half, DeferredReduction<redop<__half>>>();       \
    type<complex<float>>::                                                                         \
      register_variants_with_return<complex<float>, DeferredReduction<redop<complex<float>>>>();   \
  }
   // type<complex<double>>::register_variants_with_return<complex<double>,
   // DeferredReduction<redop<complex<double>>>>();       \

#define REGISTER_NONCOMPLEX_TASKS_WITH_REDUCTION_RETURN(type, redop)                               \
  {                                                                                                \
    type<float>::register_variants_with_return<float, DeferredReduction<redop<float>>>();          \
    type<double>::register_variants_with_return<double, DeferredReduction<redop<double>>>();       \
    type<int16_t>::register_variants_with_return<int16_t, DeferredReduction<redop<int16_t>>>();    \
    type<int32_t>::register_variants_with_return<int32_t, DeferredReduction<redop<int32_t>>>();    \
    type<int64_t>::register_variants_with_return<int64_t, DeferredReduction<redop<int64_t>>>();    \
    type<uint16_t>::register_variants_with_return<uint16_t, DeferredReduction<redop<uint16_t>>>(); \
    type<uint32_t>::register_variants_with_return<uint32_t, DeferredReduction<redop<uint32_t>>>(); \
    type<uint64_t>::register_variants_with_return<uint64_t, DeferredReduction<redop<uint64_t>>>(); \
    type<bool>::register_variants_with_return<bool, DeferredReduction<redop<bool>>>();             \
    type<__half>::register_variants_with_return<__half, DeferredReduction<redop<__half>>>();       \
  }

#define REGISTER_ALL_TASKS_WITH_WRAP_REDUCTION_RETURN(type, wrap, redop)                           \
  {                                                                                                \
    type<float>::register_variants_with_return<wrap<float>, DeferredReduction<redop<float>>>();    \
    type<double>::register_variants_with_return<wrap<double>, DeferredReduction<redop<double>>>(); \
    type<int16_t>::register_variants_with_return<wrap<int16_t>,                                    \
                                                 DeferredReduction<redop<int16_t>>>();             \
    type<int32_t>::register_variants_with_return<wrap<int32_t>,                                    \
                                                 DeferredReduction<redop<int32_t>>>();             \
    type<int64_t>::register_variants_with_return<wrap<int64_t>,                                    \
                                                 DeferredReduction<redop<int64_t>>>();             \
    type<uint16_t>::register_variants_with_return<wrap<uint16_t>,                                  \
                                                  DeferredReduction<redop<uint16_t>>>();           \
    type<uint32_t>::register_variants_with_return<wrap<uint32_t>,                                  \
                                                  DeferredReduction<redop<uint32_t>>>();           \
    type<uint64_t>::register_variants_with_return<wrap<uint64_t>,                                  \
                                                  DeferredReduction<redop<uint64_t>>>();           \
    type<bool>::register_variants_with_return<wrap<bool>, DeferredReduction<redop<bool>>>();       \
    type<__half>::register_variants_with_return<wrap<__half>, DeferredReduction<redop<__half>>>(); \
    type<complex<float>>::register_variants_with_return<                                           \
      wrap<complex<float>>,                                                                        \
      DeferredReduction<redop<complex<float>>>>();                                                 \
  }
   // type<complex<double>>::register_variants_with_return<wrap<complex<double>>,
   // DeferredReduction<redop<complex<double>>>>(); \

#define REGISTER_NONCOMPLEX_TASKS_WITH_WRAP_REDUCTION_RETURN(type, wrap, redop)                    \
  {                                                                                                \
    type<float>::register_variants_with_return<wrap<float>, DeferredReduction<redop<float>>>();    \
    type<double>::register_variants_with_return<wrap<double>, DeferredReduction<redop<double>>>(); \
    type<int16_t>::register_variants_with_return<wrap<int16_t>,                                    \
                                                 DeferredReduction<redop<int16_t>>>();             \
    type<int32_t>::register_variants_with_return<wrap<int32_t>,                                    \
                                                 DeferredReduction<redop<int32_t>>>();             \
    type<int64_t>::register_variants_with_return<wrap<int64_t>,                                    \
                                                 DeferredReduction<redop<int64_t>>>();             \
    type<uint16_t>::register_variants_with_return<wrap<uint16_t>,                                  \
                                                  DeferredReduction<redop<uint16_t>>>();           \
    type<uint32_t>::register_variants_with_return<wrap<uint32_t>,                                  \
                                                  DeferredReduction<redop<uint32_t>>>();           \
    type<uint64_t>::register_variants_with_return<wrap<uint64_t>,                                  \
                                                  DeferredReduction<redop<uint64_t>>>();           \
    type<bool>::register_variants_with_return<wrap<bool>, DeferredReduction<redop<bool>>>();       \
    type<__half>::register_variants_with_return<wrap<__half>, DeferredReduction<redop<__half>>>(); \
  }

#define REGISTER_ALL_TASKS_WITH_REDUCTION_ARG_RETURN(type, redop)                                  \
  {                                                                                                \
    type<float>::register_variants_with_return<uint64_t, DeferredReduction<redop<uint64_t>>>();    \
    type<double>::register_variants_with_return<uint64_t, DeferredReduction<redop<uint64_t>>>();   \
    type<int16_t>::register_variants_with_return<uint64_t, DeferredReduction<redop<uint64_t>>>();  \
    type<int32_t>::register_variants_with_return<uint64_t, DeferredReduction<redop<uint64_t>>>();  \
    type<int64_t>::register_variants_with_return<uint64_t, DeferredReduction<redop<uint64_t>>>();  \
    type<uint16_t>::register_variants_with_return<uint64_t, DeferredReduction<redop<uint64_t>>>(); \
    type<uint32_t>::register_variants_with_return<uint64_t, DeferredReduction<redop<uint64_t>>>(); \
    type<uint64_t>::register_variants_with_return<uint64_t, DeferredReduction<redop<uint64_t>>>(); \
    type<bool>::register_variants_with_return<uint64_t, DeferredReduction<redop<uint64_t>>>();     \
    type<__half>::register_variants_with_return<uint64_t, DeferredReduction<redop<uint64_t>>>();   \
    type<complex<float>>::register_variants_with_return<uint64_t,                                  \
                                                        DeferredReduction<redop<uint64_t>>>();     \
  }
   // type<complex<double>>::register_variants_with_return<uint64_t,
   // DeferredReduction<redop<uint64_t>>>();   \

#define REGISTER_INT_TASKS_WITH_RETURN(type)                          \
  {                                                                   \
    type<int16_t>::register_variants_with_return<int16_t, int16_t>(); \
    type<int32_t>::register_variants_with_return<int32_t, int32_t>(); \
    type<int64_t>::register_variants_with_return<int64_t, int64_t>(); \
  }

#define REGISTER_REAL_TASKS_WITH_RETURN(type)                      \
  {                                                                \
    type<float>::register_variants_with_return<float, float>();    \
    type<double>::register_variants_with_return<double, double>(); \
    type<__half>::register_variants_with_return<__half, __half>(); \
  }

#define REGISTER_ALL_REDUCTIONS(type, offset)                                                      \
  {                                                                                                \
    Runtime::register_reduction_op<type<float>>(offset + type<float>::REDOP_ID);                   \
    Runtime::register_reduction_op<type<double>>(offset + type<double>::REDOP_ID);                 \
    Runtime::register_reduction_op<type<int16_t>>(offset + type<int16_t>::REDOP_ID);               \
    Runtime::register_reduction_op<type<int32_t>>(offset + type<int32_t>::REDOP_ID);               \
    Runtime::register_reduction_op<type<int64_t>>(offset + type<int64_t>::REDOP_ID);               \
    Runtime::register_reduction_op<type<uint16_t>>(offset + type<uint16_t>::REDOP_ID);             \
    Runtime::register_reduction_op<type<uint32_t>>(offset + type<uint32_t>::REDOP_ID);             \
    Runtime::register_reduction_op<type<uint64_t>>(offset + type<uint64_t>::REDOP_ID);             \
    Runtime::register_reduction_op<type<bool>>(offset + type<bool>::REDOP_ID);                     \
    Runtime::register_reduction_op<type<__half>>(offset + type<__half>::REDOP_ID);                 \
    Runtime::register_reduction_op<type<complex<float>>>(offset + type<complex<float>>::REDOP_ID); \
  }
   // Runtime::register_reduction_op<type<complex<double>>>(offset +
   // type<complex<double>>::REDOP_ID);
