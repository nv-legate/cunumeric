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

#ifndef __NUMPY_SCALAR_UNARY_OPERATION_H__
#define __NUMPY_SCALAR_UNARY_OPERATION_H__

#include "numpy.h"
#include "point_task.h"

namespace legate {
namespace numpy {

// For doing a scalar unary operation
// XXX make this derive from PointTask
template<class Derived, typename UnaryFunction>
class ScalarUnaryOperationTask : public NumPyTask<Derived> {
private:
  using argument_type = typename UnaryFunction::argument_type;
  using result_type   = std::result_of_t<UnaryFunction(argument_type)>;

public:
  // XXX figure out how to hoist this into PointTask
  static const int TASK_ID = task_id<UnaryFunction::op_code, NUMPY_SCALAR_VARIANT_OFFSET, result_type, argument_type>;

  static const int REGIONS = 0;

  static result_type cpu_variant(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
                                 Legion::Runtime* runtime) {

    argument_type rhs = task->futures[0].get_result<argument_type>(true /*silence warnings*/);

    // return the result of the UnaryFunction
    UnaryFunction func;
    return func(rhs);
  }

private:
  struct StaticRegistrar {
    StaticRegistrar() { ScalarUnaryOperationTask::template register_variants_with_return<result_type, argument_type>(); }
  };

  virtual void force_instantiation_of_static_registrar() { (void)&static_registrar; }

  // this static member registers this task's variants during static initialization
  static const StaticRegistrar static_registrar;
};

// this is the definition of ScalarUnaryOperationTask::static_registrar
template<class Derived, class UnaryFunction>
const typename ScalarUnaryOperationTask<Derived, UnaryFunction>::StaticRegistrar
    ScalarUnaryOperationTask<Derived, UnaryFunction>::static_registrar{};

}    // namespace numpy
}    // namespace legate

#endif    // __NUMPY_SCALAR_UNARY_OPERATION_H__
