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

#include <stdio.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "cudalibs.h"

namespace cunumeric {

CUDALibraries::CUDALibraries() : cublas_(nullptr), cusolver_(nullptr) {}

CUDALibraries::~CUDALibraries() { finalize(); }

void CUDALibraries::finalize()
{
  if (cublas_ != nullptr) finalize_cublas();
  if (cusolver_ != nullptr) finalize_cusolver();
}

void CUDALibraries::finalize_cublas()
{
  cublasStatus_t status = cublasDestroy(cublas_);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,
            "Internal cuBLAS destruction failure "
            "with error code %d in cuNumeric\n",
            status);
    abort();
  }
  cublas_ = nullptr;
}

void CUDALibraries::finalize_cusolver()
{
  cusolverStatus_t status = cusolverDnDestroy(cusolver_);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    fprintf(stderr,
            "Internal cuSOLVER destruction failure "
            "with error code %d in cuNumeric\n",
            status);
    abort();
  }
  cusolver_ = nullptr;
}

cublasContext* CUDALibraries::get_cublas()
{
  if (nullptr == cublas_) {
    cublasStatus_t status = cublasCreate(&cublas_);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr,
              "Internal cuBLAS initialization failure "
              "with error code %d in cuNumeric\n",
              status);
      abort();
    }
    const char* disable_tensor_cores = getenv("LEGATE_DISABLE_TENSOR_CORES");
    if (nullptr == disable_tensor_cores) {
      // No request to disable tensor cores so turn them on
      status = cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH);
      if (status != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "WARNING: cuBLAS does not support Tensor cores!");
    }
  }
  return cublas_;
}

cusolverDnContext* CUDALibraries::get_cusolver()
{
  if (nullptr == cusolver_) {
    cusolverStatus_t status = cusolverDnCreate(&cusolver_);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      fprintf(stderr,
              "Internal cuSOLVER initialization failure "
              "with error code %d in cuNumeric\n",
              status);
      abort();
    }
  }
  return cusolver_;
}

}  // namespace cunumeric
