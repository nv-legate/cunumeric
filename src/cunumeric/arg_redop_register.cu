/* Copyright 2021-2022 NVIDIA Corporation
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

#include "cunumeric/arg_redop_register.h"

extern "C" {

void cunumeric_register_reduction_op(int32_t type_uid, int32_t _elem_type_code)
{
  auto elem_type_code = static_cast<legate::Type::Code>(_elem_type_code);
  legate::type_dispatch(elem_type_code, cunumeric::register_reduction_op_fn{}, type_uid);
}
}
