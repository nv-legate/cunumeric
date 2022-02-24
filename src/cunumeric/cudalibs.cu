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

#include <mutex>
#include <stdio.h>

namespace cunumeric {

using namespace Legion;

static Logger log_cudalibs("cunumeric.cudalibs");

cufftContext::cufftContext(cufftPlan* plan) : plan_(plan) {}

cufftContext::~cufftContext()
{
  auto hdl = handle();
  for (auto type : callback_types_) CHECK_CUFFT(cufftXtClearCallback(hdl, type));
}

cufftHandle cufftContext::handle() { return plan_->handle; }

size_t cufftContext::workarea_size() { return plan_->workarea; }

void cufftContext::set_callback(cufftXtCallbackType type, void* callback, void* data)
{
  auto hdl = handle();
  if (callback_types_.find(type) != callback_types_.end())
    CHECK_CUFFT(cufftXtClearCallback(hdl, type));
  void* callbacks[1] = {callback};
  void* datas[1]     = {data};
  CHECK_CUFFT(cufftXtSetCallback(hdl, callbacks, type, datas));
  callback_types_.insert(type);
}

struct cufftPlanCache {
 private:
  // Maximum number of plans to keep per dimension
  static constexpr int32_t MAX_PLANS = 4;

 private:
  struct LRUEntry {
    std::unique_ptr<cufftPlan> plan{nullptr};
    DomainPoint fftshape{};
    uint32_t lru_index{0};
  };

 public:
  cufftPlanCache(cufftType type);
  ~cufftPlanCache();

 public:
  cufftPlan* get_cufft_plan(const DomainPoint& size);

 private:
  using Cache = std::array<LRUEntry, MAX_PLANS>;
  std::array<Cache, LEGION_MAX_DIM + 1> cache_{};
  cufftType type_;
};

cufftPlanCache::cufftPlanCache(cufftType type) : type_(type)
{
  for (auto& cache : cache_)
    for (auto& entry : cache) assert(0 == entry.fftshape.dim);
}

cufftPlanCache::~cufftPlanCache()
{
  for (auto& cache : cache_)
    for (auto& entry : cache)
      if (entry.plan != nullptr) CHECK_CUFFT(cufftDestroy(entry.plan->handle));
}

cufftPlan* cufftPlanCache::get_cufft_plan(const DomainPoint& size)
{
  int32_t match = -1;
  auto& cache   = cache_[size.dim];
  for (int32_t idx = 0; idx < MAX_PLANS; ++idx)
    if (cache[idx].fftshape == size) {
      match = idx;
      break;
    }

  cufftPlan* result{nullptr};
  // If there's no match, we create a new plan
  if (-1 == match) {
    log_cudalibs.debug() << "[cufftPlanCache] no match found for " << size << " (type: " << type_
                         << ")";
    int32_t plan_index = -1;
    for (int32_t idx = 0; idx < MAX_PLANS; ++idx) {
      auto& entry = cache[idx];
      if (nullptr == entry.plan) {
        log_cudalibs.debug() << "[cufftPlanCache] found empty entry " << idx << " (type: " << type_
                             << ")";
        entry.plan      = std::make_unique<cufftPlan>();
        entry.lru_index = idx;
        plan_index      = idx;
        break;
      } else if (entry.lru_index == MAX_PLANS - 1) {
        log_cudalibs.debug() << "[cufftPlanCache] evict entry " << idx << " for " << entry.fftshape
                             << " (type: " << type_ << ")";
        CHECK_CUFFT(cufftDestroy(entry.plan->handle));
        plan_index = idx;
        break;
      }
    }
    assert(plan_index != -1);
    auto& entry    = cache[plan_index];
    entry.fftshape = size;
    result         = entry.plan.get();

    CHECK_CUFFT(cufftCreate(&result->handle));
    CHECK_CUFFT(cufftSetAutoAllocation(result->handle, 0 /*we'll do the allocation*/));

    std::vector<int32_t> n(size.dim);
    for (int32_t dim = 0; dim < size.dim; ++dim) n[dim] = size[dim];
    CHECK_CUFFT(cufftMakePlanMany(result->handle,
                                  size.dim,
                                  n.data(),
                                  nullptr,
                                  1,
                                  1,
                                  nullptr,
                                  1,
                                  1,
                                  type_,
                                  1 /*batch*/,
                                  &result->workarea));
  }
  // Otherwise, we return the cached plan and adjust the LRU count
  else {
    log_cudalibs.debug() << "[cufftPlanCache] found match for " << size << " (type: " << type_
                         << ")";
    auto& entry = cache[match];
    result      = entry.plan.get();

    for (int32_t idx = 0; idx < MAX_PLANS; ++idx) {
      auto& other = cache[idx];
      if (other.lru_index < entry.lru_index) ++other.lru_index;
    }
    entry.lru_index = 0;
  }
  return result;
}

CUDALibraries::CUDALibraries()
  : finalized_(false), cublas_(nullptr), cusolver_(nullptr), cutensor_(nullptr), plan_caches_()
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
  for (auto& pair : plan_caches_) delete pair.second;
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

cufftContext CUDALibraries::get_cufft_plan(cufftType type, const DomainPoint& size)
{
  auto finder = plan_caches_.find(type);
  cufftPlanCache* cache{nullptr};

  if (plan_caches_.end() == finder) {
    cache              = new cufftPlanCache(type);
    plan_caches_[type] = cache;
  } else
    cache = finder->second;
  return cufftContext(cache->get_cufft_plan(size));
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

cufftContext get_cufft_plan(cufftType type, const Legion::DomainPoint& size)
{
  const auto proc = Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cufft_plan(type, size);
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
