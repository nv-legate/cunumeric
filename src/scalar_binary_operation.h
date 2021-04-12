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

#ifndef __NUMPY_SCALAR_BINARY_OPERATION_H__
#define __NUMPY_SCALAR_BINARY_OPERATION_H__

#include "numpy.h"
#include "point_task.h"    // for legate_type_code_of

namespace legate {
namespace numpy {

// For doing a scalar binary operation
template<class Derived, typename BinaryFunction>
class ScalarBinaryOperationTask : public NumPyTask<Derived> {
private:
  using first_argument_type  = typename BinaryFunction::first_argument_type;
  using second_argument_type = typename BinaryFunction::second_argument_type;
  using result_type          = std::result_of_t<BinaryFunction(first_argument_type, second_argument_type)>;

public:
  static_assert(std::is_same<first_argument_type, second_argument_type>::value,
                "ScalarBinaryOperationTask currently requires first_argument_type and second_argument_type to be the same type.");
  static constexpr int TASK_ID =
      task_id<BinaryFunction::op_code, NUMPY_SCALAR_VARIANT_OFFSET, result_type, first_argument_type, second_argument_type>;
  static const int REGIONS = 0;

  static result_type cpu_variant(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
                                 Legion::Runtime* runtime) {
    // deserialize arguments
    first_argument_type  lhs = task->futures[0].get_result<first_argument_type>(true /*silence warnings*/);
    second_argument_type rhs = task->futures[1].get_result<second_argument_type>(true /*silence warnings*/);

    // return the result of the BinaryFunction
    BinaryFunction func;
    return func(lhs, rhs);
  }

private:
  struct StaticRegistrar {
    StaticRegistrar() { ScalarBinaryOperationTask::template register_variants_with_return<result_type, first_argument_type>(); }
  };

  virtual void force_instantiation_of_static_registrar() { (void)&static_registrar; }

  // this static member registers this task's variants during static initialization
  static const StaticRegistrar static_registrar;
};

// this is the definition of ScalarBinaryOperationTask::static_registrar
template<class Derived, class BinaryFunction>
const typename ScalarBinaryOperationTask<Derived, BinaryFunction>::StaticRegistrar
    ScalarBinaryOperationTask<Derived, BinaryFunction>::static_registrar{};

}    // namespace numpy
}    // namespace legate

#endif    // __NUMPY_SCALAR_BINARY_OPERATION_H__
