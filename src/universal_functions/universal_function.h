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

#ifndef __NUMPY_UNIVERSAL_FUNCTION_H__
#define __NUMPY_UNIVERSAL_FUNCTION_H__

#include "binary_operation.h"
#include "broadcast_binary_operation.h"
#include "inplace_binary_operation.h"
#include "inplace_broadcast_binary_operation.h"
#include "inplace_unary_operation.h"
#include "noncommutative_broadcast_binary_operation.h"
#include "scalar_binary_operation.h"
#include "scalar_unary_operation.h"
#include "unary_operation.h"
#include <functional>
#include <memory>  // unique_ptr

namespace legate {
namespace numpy {

template <class UnaryOperation>
struct UnaryUniversalFunction {
  // a unary universal function instantiates the following kinds of tasks:
  struct NormalTask : public UnaryOperationTask<NormalTask, UnaryOperation> {
  };
  struct InplaceNormalTask : public InplaceUnaryOperationTask<InplaceNormalTask, UnaryOperation> {
  };
  struct ScalarTask : public ScalarUnaryOperationTask<ScalarTask, UnaryOperation> {
  };

  void instantiate_tasks()
  {
    // this function will never be called
    // it exists simply to force the instantiation of
    // NormalTask, InplaceNormalTask, and ScalarTask

    std::unique_ptr<NormalTask> ptr1(new NormalTask);
    std::unique_ptr<InplaceNormalTask> ptr2(new InplaceNormalTask);
    std::unique_ptr<ScalarTask> ptr3(new ScalarTask);
  }

#if defined(LEGATE_USE_CUDA) and defined(__CUDACC__)
  void instantiate_task_gpu_variants()
  {
    // this function will never be called
    // it exists simply to force the .gpu_variant() members of NormalTask and InplaceNormalTask
    // to be instantiated and retained by the compiler

    printf("%p %p", NormalTask::gpu_variant, InplaceNormalTask::gpu_variant);
  }
#endif
};

template <class BinaryOperation>
struct BinaryUniversalFunction {
  // a binary universal function instantiates the following kinds of tasks:
  struct NormalTask : public BinaryOperationTask<NormalTask, BinaryOperation> {
  };
  struct InplaceNormalTask : public InplaceBinaryOperationTask<InplaceNormalTask, BinaryOperation> {
  };
  struct ScalarTask : public ScalarBinaryOperationTask<ScalarTask, BinaryOperation> {
  };
  struct BroadcastTask : public BroadcastBinaryOperationTask<BroadcastTask, BinaryOperation> {
  };
  struct InplaceBroadcastTask
    : public InplaceBroadcastBinaryOperationTask<InplaceBroadcastTask, BinaryOperation> {
  };

  void instantiate_tasks()
  {
    // this function will never be called
    // it exists simply to force the instantiation of
    // NormalTask, InplaceNormalTask, ScalarTask, BroadcastTask, and InplaceBroadcastTask

    std::unique_ptr<NormalTask> ptr1(new NormalTask);
    std::unique_ptr<InplaceNormalTask> ptr2(new InplaceNormalTask);
    std::unique_ptr<ScalarTask> ptr3(new ScalarTask);
    std::unique_ptr<BroadcastTask> ptr4(new BroadcastTask);
    std::unique_ptr<InplaceBroadcastTask> ptr5(new InplaceBroadcastTask);
  }

#if defined(LEGATE_USE_CUDA) and defined(__CUDACC__)
  void instantiate_task_gpu_variants()
  {
    // this function will never be called
    // it exists simply to force the .gpu_variant() members of these
    // tasks to be instanitated and retained by the compiler

    printf("%p %p %p %p",
           NormalTask::gpu_variant,
           InplaceNormalTask::gpu_variant,
           BroadcastTask::gpu_variant,
           InplaceBroadcastTask::gpu_variant);
  }
#endif
};

// XXX investigate whether it would be convenient to merge this class with BinaryUniversalFunction
// somehow
template <class BinaryOperation>
struct NoncommutativeBinaryUniversalFunction {
  // a binary universal function instantiates the following kinds of tasks:
  struct NormalTask : public BinaryOperationTask<NormalTask, BinaryOperation> {
  };
  struct InplaceNormalTask : public InplaceBinaryOperationTask<InplaceNormalTask, BinaryOperation> {
  };
  struct ScalarTask : public ScalarBinaryOperationTask<ScalarTask, BinaryOperation> {
  };
  struct BroadcastTask
    : public NoncommutativeBroadcastBinaryOperationTask<BroadcastTask, BinaryOperation> {
  };
  struct InplaceBroadcastTask
    : public InplaceBroadcastBinaryOperationTask<InplaceBroadcastTask, BinaryOperation> {
  };

  void instantiate_tasks()
  {
    // this function will never be called
    // it exists simply to force the instantiation of
    // NormalTask, InplaceNormalTask, ScalarTask, BroadcastTask, and InplaceBroadcastTask

    std::unique_ptr<NormalTask> ptr1(new NormalTask);
    std::unique_ptr<InplaceNormalTask> ptr2(new InplaceNormalTask);
    std::unique_ptr<ScalarTask> ptr3(new ScalarTask);
    std::unique_ptr<BroadcastTask> ptr4(new BroadcastTask);
    std::unique_ptr<InplaceBroadcastTask> ptr5(new InplaceBroadcastTask);
  }

#if defined(LEGATE_USE_CUDA) and defined(__CUDACC__)
  void instantiate_task_gpu_variants()
  {
    // this function will never be called
    // it exists simply to force the .gpu_variant() members of these
    // tasks to be instanitated and retained by the compiler

    printf("%p %p %p %p",
           NormalTask::gpu_variant,
           InplaceNormalTask::gpu_variant,
           BroadcastTask::gpu_variant,
           InplaceBroadcastTask::gpu_variant);
  }
#endif
};

}  // namespace numpy
}  // namespace legate

#endif  // __NUMPY_UNIVERSAL_FUNCTION_H__
