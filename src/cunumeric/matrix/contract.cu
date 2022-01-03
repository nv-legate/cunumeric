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

#include "cunumeric/matrix/contract.h"
#include "cunumeric/matrix/contract_template.inl"

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;

template <cudaDataType_t DATA_TYPE_CODE, cutensorComputeType_t COMPUTE_TYPE_CODE, typename T>
__host__ void contract(T* lhs_data,
                       size_t lhs_ndim,
                       int64_t* lhs_shape,
                       int64_t* lhs_strides,
                       int32_t* lhs_modes,
                       const T* rhs1_data,
                       size_t rhs1_ndim,
                       int64_t* rhs1_shape,
                       int64_t* rhs1_strides,
                       int32_t* rhs1_modes,
                       const T* rhs2_data,
                       size_t rhs2_ndim,
                       int64_t* rhs2_shape,
                       int64_t* rhs2_strides,
                       int32_t* rhs2_modes)
{
  // Initialization
  cudaStream_t task_stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&task_stream, cudaStreamNonBlocking));
  cutensorHandle_t* handle = get_cutensor();

  // Create tensor descriptors
  cutensorTensorDescriptor_t lhs_desc;
  cutensorTensorDescriptor_t rhs1_desc;
  cutensorTensorDescriptor_t rhs2_desc;
  CHECK_CUTENSOR(cutensorInitTensorDescriptor(
    handle, &lhs_desc, lhs_ndim, lhs_shape, lhs_strides, DATA_TYPE_CODE, CUTENSOR_OP_IDENTITY));
  CHECK_CUTENSOR(cutensorInitTensorDescriptor(
    handle, &rhs1_desc, rhs1_ndim, rhs1_shape, rhs1_strides, DATA_TYPE_CODE, CUTENSOR_OP_IDENTITY));
  CHECK_CUTENSOR(cutensorInitTensorDescriptor(
    handle, &rhs2_desc, rhs2_ndim, rhs2_shape, rhs2_strides, DATA_TYPE_CODE, CUTENSOR_OP_IDENTITY));

  // Prepare algorithm description
  uint32_t lhs_req;
  uint32_t rhs1_req;
  uint32_t rhs2_req;
  CHECK_CUTENSOR(cutensorGetAlignmentRequirement(handle, lhs_data, &lhs_desc, &lhs_req));
  CHECK_CUTENSOR(cutensorGetAlignmentRequirement(handle, rhs1_data, &rhs1_desc, &rhs1_req));
  CHECK_CUTENSOR(cutensorGetAlignmentRequirement(handle, rhs2_data, &rhs2_desc, &rhs2_req));
  cutensorContractionDescriptor_t desc;
  CHECK_CUTENSOR(cutensorInitContractionDescriptor(handle,
                                                   &desc,
                                                   &rhs1_desc,
                                                   rhs1_modes,
                                                   rhs1_req,
                                                   &rhs2_desc,
                                                   rhs2_modes,
                                                   rhs2_req,
                                                   &lhs_desc,
                                                   lhs_modes,
                                                   lhs_req,
                                                   &lhs_desc,
                                                   lhs_modes,
                                                   lhs_req,
                                                   COMPUTE_TYPE_CODE));
  cutensorContractionFind_t find;
  CHECK_CUTENSOR(cutensorInitContractionFind(handle, &find, CUTENSOR_ALGO_DEFAULT));

  // Allocate intermediate storage
  uint64_t work_size = 0;
  CHECK_CUTENSOR(cutensorContractionGetWorkspace(
    handle, &desc, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &work_size));
  DeferredBuffer<uint8_t, 1> work_buf(Rect<1>(Point<1>(0), Point<1>(work_size - 1)),
                                      Memory::GPU_FB_MEM);
  void* work = work_buf.ptr(Point<1>(0));

  // Execute contraction
  cutensorContractionPlan_t plan;
  CHECK_CUTENSOR(cutensorInitContractionPlan(handle, &plan, &desc, &find, work_size));
  const T alpha = 1.0;
  const T beta  = 0.0;
  CHECK_CUTENSOR(cutensorContraction(handle,
                                     &plan,
                                     &alpha,
                                     rhs1_data,
                                     rhs2_data,
                                     &beta,
                                     lhs_data,
                                     lhs_data,
                                     work,
                                     work_size,
                                     task_stream));

  // Cleanup
  CHECK_CUDA(cudaStreamDestroy(task_stream));
}

template <>
struct ContractImplBody<VariantKind::GPU, LegateTypeCode::FLOAT_LT> {
  void operator()(float* lhs_data,
                  size_t lhs_ndim,
                  int64_t* lhs_shape,
                  int64_t* lhs_strides,
                  int32_t* lhs_modes,
                  const float* rhs1_data,
                  size_t rhs1_ndim,
                  int64_t* rhs1_shape,
                  int64_t* rhs1_strides,
                  int32_t* rhs1_modes,
                  const float* rhs2_data,
                  size_t rhs2_ndim,
                  int64_t* rhs2_shape,
                  int64_t* rhs2_strides,
                  int32_t* rhs2_modes)
  {
    contract<CUDA_R_32F, CUTENSOR_COMPUTE_32F>(lhs_data,
                                               lhs_ndim,
                                               lhs_shape,
                                               lhs_strides,
                                               lhs_modes,
                                               rhs1_data,
                                               rhs1_ndim,
                                               rhs1_shape,
                                               rhs1_strides,
                                               rhs1_modes,
                                               rhs2_data,
                                               rhs2_ndim,
                                               rhs2_shape,
                                               rhs2_strides,
                                               rhs2_modes);
  }
};

template <>
struct ContractImplBody<VariantKind::GPU, LegateTypeCode::DOUBLE_LT> {
  void operator()(double* lhs_data,
                  size_t lhs_ndim,
                  int64_t* lhs_shape,
                  int64_t* lhs_strides,
                  int32_t* lhs_modes,
                  const double* rhs1_data,
                  size_t rhs1_ndim,
                  int64_t* rhs1_shape,
                  int64_t* rhs1_strides,
                  int32_t* rhs1_modes,
                  const double* rhs2_data,
                  size_t rhs2_ndim,
                  int64_t* rhs2_shape,
                  int64_t* rhs2_strides,
                  int32_t* rhs2_modes)
  {
    contract<CUDA_R_64F, CUTENSOR_COMPUTE_64F>(lhs_data,
                                               lhs_ndim,
                                               lhs_shape,
                                               lhs_strides,
                                               lhs_modes,
                                               rhs1_data,
                                               rhs1_ndim,
                                               rhs1_shape,
                                               rhs1_strides,
                                               rhs1_modes,
                                               rhs2_data,
                                               rhs2_ndim,
                                               rhs2_shape,
                                               rhs2_strides,
                                               rhs2_modes);
  }
};

template <>
struct ContractImplBody<VariantKind::GPU, LegateTypeCode::COMPLEX64_LT> {
  void operator()(complex<float>* lhs_data,
                  size_t lhs_ndim,
                  int64_t* lhs_shape,
                  int64_t* lhs_strides,
                  int32_t* lhs_modes,
                  const complex<float>* rhs1_data,
                  size_t rhs1_ndim,
                  int64_t* rhs1_shape,
                  int64_t* rhs1_strides,
                  int32_t* rhs1_modes,
                  const complex<float>* rhs2_data,
                  size_t rhs2_ndim,
                  int64_t* rhs2_shape,
                  int64_t* rhs2_strides,
                  int32_t* rhs2_modes)
  {
    contract<CUDA_C_32F, CUTENSOR_COMPUTE_32F>(lhs_data,
                                               lhs_ndim,
                                               lhs_shape,
                                               lhs_strides,
                                               lhs_modes,
                                               rhs1_data,
                                               rhs1_ndim,
                                               rhs1_shape,
                                               rhs1_strides,
                                               rhs1_modes,
                                               rhs2_data,
                                               rhs2_ndim,
                                               rhs2_shape,
                                               rhs2_strides,
                                               rhs2_modes);
  }
};

template <>
struct ContractImplBody<VariantKind::GPU, LegateTypeCode::COMPLEX128_LT> {
  void operator()(complex<double>* lhs_data,
                  size_t lhs_ndim,
                  int64_t* lhs_shape,
                  int64_t* lhs_strides,
                  int32_t* lhs_modes,
                  const complex<double>* rhs1_data,
                  size_t rhs1_ndim,
                  int64_t* rhs1_shape,
                  int64_t* rhs1_strides,
                  int32_t* rhs1_modes,
                  const complex<double>* rhs2_data,
                  size_t rhs2_ndim,
                  int64_t* rhs2_shape,
                  int64_t* rhs2_strides,
                  int32_t* rhs2_modes)
  {
    contract<CUDA_C_64F, CUTENSOR_COMPUTE_64F>(lhs_data,
                                               lhs_ndim,
                                               lhs_shape,
                                               lhs_strides,
                                               lhs_modes,
                                               rhs1_data,
                                               rhs1_ndim,
                                               rhs1_shape,
                                               rhs1_strides,
                                               rhs1_modes,
                                               rhs2_data,
                                               rhs2_ndim,
                                               rhs2_shape,
                                               rhs2_strides,
                                               rhs2_modes);
  }
};

/*static*/ void ContractTask::gpu_variant(TaskContext& context)
{
  contract_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
