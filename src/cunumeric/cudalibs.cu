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

#include "cunumeric.h"

#include "cudalibs.h"
#include "cuda_help.h"

#include <mutex>
#include <stdio.h>

namespace cunumeric {

using namespace Legion;

CUDALibraries::CUDALibraries()
  : finalized_(false), cublas_(nullptr), cusolver_(nullptr), cutensor_(nullptr)
{
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

CUDALibraries::~CUDALibraries() { finalize(); }

void CUDALibraries::finalize()
{
  if (finalized_) return;
  if (cublas_ != nullptr) finalize_cublas();
  if (cusolver_ != nullptr) finalize_cusolver();
  if (cutensor_ != nullptr) finalize_cutensor();
  cudaStreamDestroy(stream_);
  finalized_ = true;
}

void CUDALibraries::finalize_cublas()
{
  CHECK_CUBLAS(cublasDestroy(cublas_));
  cublas_ = nullptr;
}

void CUDALibraries::finalize_cusolver()
{
  CHECK_CUSOLVER(cusolverDnDestroy(cusolver_));
  cusolver_ = nullptr;
}

void CUDALibraries::finalize_cutensor()
{
  delete cutensor_;
  cutensor_ = nullptr;
}

cudaStream_t CUDALibraries::get_cached_stream() { return stream_; }

cublasHandle_t CUDALibraries::get_cublas()
{
  if (nullptr == cublas_) {
    CHECK_CUBLAS(cublasCreate(&cublas_));
    const char* disable_tensor_cores = getenv("CUNUMERIC_DISABLE_TENSOR_CORES");
    if (nullptr == disable_tensor_cores) {
      // No request to disable tensor cores so turn them on
      cublasStatus_t status = cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH);
      if (status != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "WARNING: cuBLAS does not support Tensor cores!");
    }
  }
  return cublas_;
}

cusolverDnHandle_t CUDALibraries::get_cusolver()
{
  if (nullptr == cusolver_) CHECK_CUSOLVER(cusolverDnCreate(&cusolver_));
  return cusolver_;
}

cutensorHandle_t* CUDALibraries::get_cutensor()
{
  if (nullptr == cutensor_) {
    cutensor_ = new cutensorHandle_t;
    CHECK_CUTENSOR(cutensorInit(cutensor_));
  }
  return cutensor_;
}

static CUDALibraries& get_cuda_libraries(Processor proc)
{
  if (proc.kind() != Processor::TOC_PROC) {
    fprintf(stderr, "Illegal request for CUDA libraries for non-GPU processor");
    LEGATE_ABORT;
  }
  static std::mutex mut_cuda_libraries;
  static std::map<Processor, CUDALibraries> cuda_libraries;

  std::lock_guard<std::mutex> guard(mut_cuda_libraries);

  auto finder = cuda_libraries.find(proc);
  if (finder != cuda_libraries.end())
    return finder->second;
  else
    return cuda_libraries[proc];
}

cudaStream_t get_cached_stream()
{
  const auto proc = Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cached_stream();
}

cublasContext* get_cublas()
{
  const auto proc = Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cublas();
}

cusolverDnContext* get_cusolver()
{
  const auto proc = Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cusolver();
}

cutensorHandle_t* get_cutensor()
{
  const auto proc = Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cutensor();
}

class LoadCUDALibsTask : public CuNumericTask<LoadCUDALibsTask> {
 public:
  static const int TASK_ID = CUNUMERIC_LOAD_CUDALIBS;

 public:
  static void gpu_variant(legate::TaskContext& context)
  {
    const auto proc = Processor::get_executing_processor();
    auto& lib       = get_cuda_libraries(proc);
    lib.get_cublas();
    lib.get_cusolver();
    lib.get_cutensor();
  }
};

class UnloadCUDALibsTask : public CuNumericTask<UnloadCUDALibsTask> {
 public:
  static const int TASK_ID = CUNUMERIC_UNLOAD_CUDALIBS;

 public:
  static void gpu_variant(legate::TaskContext& context)
  {
    const auto proc = Processor::get_executing_processor();
    auto& lib       = get_cuda_libraries(proc);
    lib.finalize();
  }
};

static void __attribute__((constructor)) register_tasks(void)
{
  LoadCUDALibsTask::register_variants();
  UnloadCUDALibsTask::register_variants();
}

}  // namespace cunumeric
