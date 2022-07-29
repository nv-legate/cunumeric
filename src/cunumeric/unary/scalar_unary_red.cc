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

#include "cunumeric/unary/scalar_unary_red.h"
#include "cunumeric/unary/scalar_unary_red_template.inl"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <>
struct ScalarUnaryRedImplBody<VariantKind::CPU> {
  template <class AccessorRD, class Kernel, class LHS>
  void operator()(AccessorRD& out, size_t volume, const LHS& identity, Kernel&& kernel)
  {
    auto result = identity;
    for (size_t idx = 0; idx < volume; ++idx) { kernel(result, idx); }
    out.reduce(0, result);
  }
};

/*static*/ void ScalarUnaryRedTask::cpu_variant(TaskContext& context)
{
  scalar_unary_red_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  ScalarUnaryRedTask::register_variants();
}
}  // namespace

}  // namespace cunumeric
