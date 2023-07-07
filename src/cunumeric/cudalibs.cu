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

#include <stdio.h>

using namespace legate;

namespace cunumeric {

static Logger log_cudalibs("cunumeric.cudalibs");

cufftContext::cufftContext(cufftPlan* plan) : plan_(plan) {}

cufftContext::~cufftContext()
{
  auto& hdl = handle();
  for (auto type : callback_types_) CHECK_CUFFT(cufftXtClearCallback(hdl, type));
  CHECK_CUFFT(cufftSetWorkArea(hdl, nullptr));
}

cufftHandle& cufftContext::handle() { return plan_->handle; }

size_t cufftContext::workareaSize() { return plan_->workarea_size; }

void cufftContext::setCallback(cufftXtCallbackType type, void* callback, void* data)
{
  void* callbacks[] = {callback};
  void* datas[]     = {data};
  CHECK_CUFFT(cufftXtSetCallback(handle(), callbacks, type, datas));
}

cufftPlanParams::cufftPlanParams(const DomainPoint& size)
  : rank(size.dim),
    n{0},
    inembed{0},
    onembed{0},
    istride(1),
    idist(1),
    ostride(1),
    odist(1),
    batch(1)
{
  for (int dim = 0; dim < rank; ++dim) n[dim] = size[dim];
}

cufftPlanParams::cufftPlanParams(int rank,
                                 long long int* n_,
                                 long long int* inembed_,
                                 long long int istride,
                                 long long int idist,
                                 long long int* onembed_,
                                 long long int ostride,
                                 long long int odist,
                                 long long int batch)
  : rank(rank), istride(istride), idist(idist), ostride(ostride), odist(odist), batch(batch)
{
  for (int dim = 0; dim < rank; ++dim) {
    n[dim]       = n_[dim];
    inembed[dim] = inembed_[dim];
    onembed[dim] = onembed_[dim];
  }
}

bool cufftPlanParams::operator==(const cufftPlanParams& other) const
{
  bool equal = rank == other.rank && istride == other.istride && idist == other.idist &&
               ostride == other.ostride && odist == other.odist && batch == other.batch;
  if (equal) {
    for (int dim = 0; dim < rank; ++dim) {
      equal = equal && (n[dim] == other.n[dim]);
      equal = equal && (inembed[dim] == other.inembed[dim]);
      equal = equal && (onembed[dim] == other.onembed[dim]);
      if (!equal) break;
    }
  }
  return equal;
}

std::string cufftPlanParams::to_string() const
{
  std::ostringstream ss;
  ss << "cufftPlanParams[rank(" << rank << "), n(" << n[0];
  for (int i = 1; i < rank; ++i) ss << "," << n[i];
  ss << "), inembed(" << inembed[0];
  for (int i = 1; i < rank; ++i) ss << "," << inembed[i];
  ss << "), istride(" << istride << "), idist(" << idist << "), onembed(" << onembed[0];
  for (int i = 1; i < rank; ++i) ss << "," << onembed[i];
  ss << "), ostride(" << ostride << "), odist(" << odist << "), batch(" << batch << ")]";
  return std::move(ss).str();
}

struct cufftPlanCache {
 private:
  // Maximum number of plans to keep per dimension
  static constexpr int32_t MAX_PLANS = 4;

 private:
  struct LRUEntry {
    std::unique_ptr<cufftPlan> plan{nullptr};
    std::unique_ptr<cufftPlanParams> params{nullptr};
    uint32_t lru_index{0};
  };

 public:
  cufftPlanCache(cufftType type);
  ~cufftPlanCache();

 public:
  cufftPlan* get_cufft_plan(const cufftPlanParams& params);

 private:
  using Cache = std::array<LRUEntry, MAX_PLANS>;
  std::array<Cache, LEGATE_MAX_DIM + 1> cache_{};
  cufftType type_;
  int64_t cache_hits_{0};
  int64_t cache_requests_{0};
};

cufftPlanCache::cufftPlanCache(cufftType type) : type_(type)
{
  for (auto& cache : cache_)
    for (auto& entry : cache) assert(0 == entry.lru_index);
}

cufftPlanCache::~cufftPlanCache()
{
  for (auto& cache : cache_)
    for (auto& entry : cache)
      if (entry.plan != nullptr) CHECK_CUFFT(cufftDestroy(entry.plan->handle));
}

cufftPlan* cufftPlanCache::get_cufft_plan(const cufftPlanParams& params)
{
  cache_requests_++;
  int32_t match = -1;
  auto& cache   = cache_[params.rank];
  for (int32_t idx = 0; idx < MAX_PLANS; ++idx) {
    auto& entry = cache[idx];
    if (nullptr == entry.plan) break;
    if (*entry.params == params) {
      match = idx;
      cache_hits_++;
      break;
    }
  }

  float hit_rate = static_cast<float>(cache_hits_) / cache_requests_;

  cufftPlan* result{nullptr};
  // If there's no match, we create a new plan
  if (-1 == match) {
    log_cudalibs.debug() << "[cufftPlanCache] no match found for " << params.to_string()
                         << " (type: " << type_ << ", hitrate: " << hit_rate << ")";
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
        log_cudalibs.debug() << "[cufftPlanCache] evict entry " << idx << " for "
                             << entry.params->to_string() << " (type: " << type_ << ")";
        CHECK_CUFFT(cufftDestroy(entry.plan->handle));
        plan_index = idx;
        // create new plan
        entry.plan = std::make_unique<cufftPlan>();
        break;
      } else {
        entry.lru_index++;
      }
    }
    assert(plan_index != -1);
    auto& entry = cache[plan_index];

    if (entry.lru_index != 0) {
      for (int32_t idx = plan_index + 1; idx < MAX_PLANS; ++idx) {
        auto& other = cache[idx];
        if (nullptr == other.plan) break;
        ++other.lru_index;
      }
      entry.lru_index = 0;
    }

    entry.params = std::make_unique<cufftPlanParams>(params);
    result       = entry.plan.get();

    auto stream = get_cached_stream();
    CHECK_CUFFT(cufftCreate(&result->handle));
    CHECK_CUFFT(cufftSetAutoAllocation(result->handle, 0 /*we'll do the allocation*/));
    // this should always be the correct stream, as we have a cache per GPU-proc
    CHECK_CUFFT(cufftSetStream(result->handle, stream));
    CHECK_CUFFT(cufftMakePlanMany64(result->handle,
                                    entry.params->rank,
                                    entry.params->n,
                                    entry.params->inembed[0] != 0 ? entry.params->inembed : nullptr,
                                    entry.params->istride,
                                    entry.params->idist,
                                    entry.params->onembed[0] != 0 ? entry.params->onembed : nullptr,
                                    entry.params->ostride,
                                    entry.params->odist,
                                    type_,
                                    entry.params->batch,
                                    &result->workarea_size));

  }
  // Otherwise, we return the cached plan and adjust the LRU count
  else {
    log_cudalibs.debug() << "[cufftPlanCache] found match for " << params.to_string()
                         << " (type: " << type_ << ", hitrate: " << hit_rate << ")";
    auto& entry = cache[match];
    result      = entry.plan.get();

    if (entry.lru_index != 0) {
      for (int32_t idx = 0; idx < MAX_PLANS; ++idx) {
        auto& other = cache[idx];
        if (other.lru_index < entry.lru_index) ++other.lru_index;
      }
      entry.lru_index = 0;
    }
  }
  return result;
}

CUDALibraries::CUDALibraries()
  : finalized_(false), cublas_(nullptr), cusolver_(nullptr), cutensor_(nullptr), plan_caches_()
{
}

CUDALibraries::~CUDALibraries() { finalize(); }

void CUDALibraries::finalize()
{
  if (finalized_) return;
  if (cublas_ != nullptr) finalize_cublas();
  if (cusolver_ != nullptr) finalize_cusolver();
  if (cutensor_ != nullptr) finalize_cutensor();
  for (auto& pair : plan_caches_) delete pair.second;
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

cublasHandle_t CUDALibraries::get_cublas()
{
  if (nullptr == cublas_) {
    CHECK_CUBLAS(cublasCreate(&cublas_));
    const char* fast_math = getenv("CUNUMERIC_FAST_MATH");
    if (fast_math != nullptr && atoi(fast_math) > 0) {
      // Enable acceleration of single precision routines using TF32 tensor cores.
      cublasStatus_t status = cublasSetMathMode(cublas_, CUBLAS_TF32_TENSOR_OP_MATH);
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

cufftContext CUDALibraries::get_cufft_plan(cufftType type, const cufftPlanParams& params)
{
  auto finder = plan_caches_.find(type);
  cufftPlanCache* cache{nullptr};

  if (plan_caches_.end() == finder) {
    cache              = new cufftPlanCache(type);
    plan_caches_[type] = cache;
  } else
    cache = finder->second;
  return cufftContext(cache->get_cufft_plan(params));
}

static CUDALibraries& get_cuda_libraries(legate::Processor proc)
{
  if (proc.kind() != legate::Processor::TOC_PROC) {
    fprintf(stderr, "Illegal request for CUDA libraries for non-GPU processor");
    LEGATE_ABORT;
  }

  static CUDALibraries cuda_libraries[LEGION_MAX_NUM_PROCS];
  const auto proc_id = proc.id & (LEGION_MAX_NUM_PROCS - 1);
  return cuda_libraries[proc_id];
}

legate::cuda::StreamView get_cached_stream()
{
  return legate::cuda::StreamPool::get_stream_pool().get_stream();
}

cublasContext* get_cublas()
{
  const auto proc = legate::Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cublas();
}

cusolverDnContext* get_cusolver()
{
  const auto proc = legate::Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cusolver();
}

cutensorHandle_t* get_cutensor()
{
  const auto proc = legate::Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cutensor();
}

cufftContext get_cufft_plan(cufftType type, const cufftPlanParams& params)
{
  const auto proc = legate::Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cufft_plan(type, params);
}

class LoadCUDALibsTask : public CuNumericTask<LoadCUDALibsTask> {
 public:
  static const int TASK_ID = CUNUMERIC_LOAD_CUDALIBS;

 public:
  static void gpu_variant(legate::TaskContext& context)
  {
    const auto proc = legate::Processor::get_executing_processor();
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
    const auto proc = legate::Processor::get_executing_processor();
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
