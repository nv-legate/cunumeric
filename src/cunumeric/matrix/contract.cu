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

#include "cunumeric/matrix/contract.h"
#include "cunumeric/matrix/contract_template.inl"

#include "cunumeric/cuda_help.h"

namespace cunumeric {

namespace {  // anonymous

template <typename T>
struct contract_helper {};

template <>
struct contract_helper<__half> {
  static const cudaDataType_t data_type_code           = CUDA_R_16F;
  static const cutensorComputeType_t compute_type_code = CUTENSOR_COMPUTE_32F;
  using scalar_t                                       = float;
};

template <>
struct contract_helper<float> {
  static const cudaDataType_t data_type_code           = CUDA_R_32F;
  static const cutensorComputeType_t compute_type_code = CUTENSOR_COMPUTE_32F;
  using scalar_t                                       = float;
};

template <>
struct contract_helper<double> {
  static const cudaDataType_t data_type_code           = CUDA_R_64F;
  static const cutensorComputeType_t compute_type_code = CUTENSOR_COMPUTE_64F;
  using scalar_t                                       = double;
};

template <>
struct contract_helper<complex<float>> {
  static const cudaDataType_t data_type_code           = CUDA_C_32F;
  static const cutensorComputeType_t compute_type_code = CUTENSOR_COMPUTE_32F;
  using scalar_t                                       = complex<float>;
};

template <>
struct contract_helper<complex<double>> {
  static const cudaDataType_t data_type_code           = CUDA_C_64F;
  static const cutensorComputeType_t compute_type_code = CUTENSOR_COMPUTE_64F;
  using scalar_t                                       = complex<double>;
};

}  // anonymous namespace

template <typename T>
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
                       int32_t* rhs2_modes,
                       bool lhs_overwritable)
{
  // Initialization
  auto handle      = get_cutensor();
  auto task_stream = get_cached_stream();

  // Create tensor descriptors
  cudaDataType_t data_type_code = contract_helper<T>::data_type_code;
  cutensorTensorDescriptor_t lhs_desc;
  cutensorTensorDescriptor_t rhs1_desc;
  cutensorTensorDescriptor_t rhs2_desc;
  CHECK_CUTENSOR(cutensorInitTensorDescriptor(
    handle, &lhs_desc, lhs_ndim, lhs_shape, lhs_strides, data_type_code, CUTENSOR_OP_IDENTITY));
  CHECK_CUTENSOR(cutensorInitTensorDescriptor(
    handle, &rhs1_desc, rhs1_ndim, rhs1_shape, rhs1_strides, data_type_code, CUTENSOR_OP_IDENTITY));
  CHECK_CUTENSOR(cutensorInitTensorDescriptor(
    handle, &rhs2_desc, rhs2_ndim, rhs2_shape, rhs2_strides, data_type_code, CUTENSOR_OP_IDENTITY));

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
                                                   contract_helper<T>::compute_type_code));
  cutensorContractionFind_t find;
  CHECK_CUTENSOR(cutensorInitContractionFind(handle, &find, CUTENSOR_ALGO_DEFAULT));

  // Allocate intermediate storage
  uint64_t work_size = 0;
  CHECK_CUTENSOR(cutensorContractionGetWorkspace(
    handle, &desc, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &work_size));
  auto work_buf = create_buffer<int8_t>(work_size, legate::Memory::GPU_FB_MEM);
  void* work    = work_buf.ptr(Point<1>(0));

  // Execute contraction
  cutensorContractionPlan_t plan;
  CHECK_CUTENSOR(cutensorInitContractionPlan(handle, &plan, &desc, &find, work_size));
  const typename contract_helper<T>::scalar_t alpha = 1.0;
  // lhs_overwritable being true means that the contraciton tasks can overwrite the lhs
  const typename contract_helper<T>::scalar_t beta = lhs_overwritable ? 0.0 : 1.0;
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

  CHECK_CUDA_STREAM(task_stream);
}

template <>
struct ContractImplBody<VariantKind::GPU, Type::Code::FLOAT16> {
  void operator()(__half* lhs_data,
                  size_t lhs_ndim,
                  int64_t* lhs_shape,
                  int64_t* lhs_strides,
                  int32_t* lhs_modes,
                  const __half* rhs1_data,
                  size_t rhs1_ndim,
                  int64_t* rhs1_shape,
                  int64_t* rhs1_strides,
                  int32_t* rhs1_modes,
                  const __half* rhs2_data,
                  size_t rhs2_ndim,
                  int64_t* rhs2_shape,
                  int64_t* rhs2_strides,
                  int32_t* rhs2_modes,
                  bool lhs_overwritable)
  {
    contract(lhs_data,
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
             rhs2_modes,
             lhs_overwritable);
  }
};

template <>
struct ContractImplBody<VariantKind::GPU, Type::Code::FLOAT32> {
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
                  int32_t* rhs2_modes,
                  bool lhs_overwritable)
  {
    contract(lhs_data,
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
             rhs2_modes,
             lhs_overwritable);
  }
};

template <>
struct ContractImplBody<VariantKind::GPU, Type::Code::FLOAT64> {
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
                  int32_t* rhs2_modes,
                  bool lhs_overwritable)
  {
    contract(lhs_data,
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
             rhs2_modes,
             lhs_overwritable);
  }
};

template <>
struct ContractImplBody<VariantKind::GPU, Type::Code::COMPLEX64> {
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
                  int32_t* rhs2_modes,
                  bool lhs_overwritable)
  {
    contract(lhs_data,
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
             rhs2_modes,
             lhs_overwritable);
  }
};

template <>
struct ContractImplBody<VariantKind::GPU, Type::Code::COMPLEX128> {
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
                  int32_t* rhs2_modes,
                  bool lhs_overwritable)
  {
    contract(lhs_data,
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
             rhs2_modes,
             lhs_overwritable);
  }
};

/*static*/ void ContractTask::gpu_variant(TaskContext& context)
{
  contract_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
