/* Copyright 2023 NVIDIA Corporation
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
#include "cunumeric/cunumeric_c.h"
#include "cunumeric/arg.h"

namespace cunumeric {

struct register_reduction_op_fn {
  template <legate::Type::Code CODE, std::enable_if_t<!legate::is_complex<CODE>::value>* = nullptr>
  void operator()(int32_t type_uid)
  {
    using VAL = legate::legate_type_of<CODE>;

    auto runtime = legate::Runtime::get_runtime();
    auto context = runtime->find_library("cunumeric");
    {
      auto redop_id =
        context->register_reduction_operator<ArgmaxReduction<VAL>>(next_reduction_operator_id());
      auto op_kind = static_cast<int32_t>(legate::ReductionOpKind::MAX);
      runtime->record_reduction_operator(type_uid, op_kind, redop_id);
    }
    {
      auto redop_id =
        context->register_reduction_operator<ArgminReduction<VAL>>(next_reduction_operator_id());
      auto op_kind = static_cast<int32_t>(legate::ReductionOpKind::MIN);
      runtime->record_reduction_operator(type_uid, op_kind, redop_id);
    }
  }

  template <legate::Type::Code CODE, std::enable_if_t<legate::is_complex<CODE>::value>* = nullptr>
  void operator()(int32_t type_uid)
  {
    LEGATE_ABORT;
  }

  static int32_t next_reduction_operator_id();
};

}  // namespace cunumeric
