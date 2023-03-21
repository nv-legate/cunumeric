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
  auto hdl = handle();
  for (auto type : callback_types_) CHECK_CUFFT(cufftXtClearCallback(hdl, type));
}

cufftHandle cufftContext::handle() { return plan_->handle; }

size_t cufftContext::workareaSize() { return plan_->workarea_size; }

void cufftContext::setCallback(cufftXtCallbackType type, void* callback, void* data)
{
  void* callbacks[] = {callback};
  void* datas[]     = {data};
  CHECK_CUFFT(cufftXtSetCallback(handle(), callbacks, type, datas));
}

cufftPlanParms::cufftPlanParms() : rank_(0) {}

cufftPlanParms::cufftPlanParms(const DomainPoint& size)
  : rank_(size.dim),
    n_{0},
    inembed_{0},
    onembed_{0},
    istride_(1),
    idist_(1),
    ostride_(1),
    odist_(1),
    batch_(1)
{
  for (int dim = 0; dim < rank_; ++dim) n_[dim] = size[dim];
}

cufftPlanParms::cufftPlanParms(int rank,
                               long long int* n,
                               long long int* inembed,
                               long long int istride,
                               long long int idist,
                               long long int* onembed,
                               long long int ostride,
                               long long int odist,
                               long long int batch)
  : rank_(rank),
    n_{0},
    inembed_{0},
    onembed_{0},
    istride_(istride),
    idist_(idist),
    ostride_(ostride),
    odist_(odist),
    batch_(batch)
{
  for (int dim = 0; dim < rank_; ++dim) {
    n_[dim]       = n[dim];
    inembed_[dim] = inembed[dim];
    onembed_[dim] = onembed[dim];
  }
}

bool cufftPlanParms::operator==(const cufftPlanParms& other) const
{
  bool equal = rank_ == other.rank_ && istride_ == other.istride_ && idist_ == other.idist_ &&
               ostride_ == other.ostride_ && odist_ == other.odist_ && batch_ == other.batch_;
  if (equal) {
    for (int dim = 0; dim < rank_; ++dim) {
      equal = equal && (n_[dim] == other.n_[dim]);
      equal = equal && (inembed_[dim] == other.inembed_[dim]);
      equal = equal && (onembed_[dim] == other.onembed_[dim]);
      if (!equal) break;
    }
  }
  return equal;
}

std::string cufftPlanParms::to_string() const
{
  std::ostringstream ss;
  ss << "cufftPlanParms[rank(" << rank_ << "), n(" << n_[0] << "), inembed(" << inembed_[0]
     << "), istride(" << istride_ << "), idist(" << idist_ << "), onembed(" << onembed_[0]
     << "), ostride(" << ostride_ << "), odist(" << odist_ << "), batch(" << batch_ << ")]";
  return std::move(ss).str();
}

struct cufftPlanCache {
 private:
  // Maximum number of plans to keep per dimension
  static constexpr int32_t MAX_PLANS = 4;

 private:
  struct LRUEntry {
    std::unique_ptr<cufftPlan> plan{nullptr};
    cufftPlanParms parms;
    uint32_t lru_index{0};
  };

 public:
  cufftPlanCache(cufftType type);
  ~cufftPlanCache();

 public:
  cufftPlan* get_cufft_plan(const cufftPlanParms& parms);

 private:
  using Cache = std::array<LRUEntry, MAX_PLANS>;
  std::array<Cache, LEGATE_MAX_DIM + 1> cache_{};
  cufftType type_;
  int64_t cache_hits_     = 0;
  int64_t cache_requests_ = 0;
};

cufftPlanCache::cufftPlanCache(cufftType type) : type_(type)
{
  for (auto& cache : cache_)
    for (auto& entry : cache) assert(0 == entry.parms.rank_);
}

cufftPlanCache::~cufftPlanCache()
{
  for (auto& cache : cache_)
    for (auto& entry : cache)
      if (entry.plan != nullptr) CHECK_CUFFT(cufftDestroy(entry.plan->handle));
}

cufftPlan* cufftPlanCache::get_cufft_plan(const cufftPlanParms& parms)
{
  cache_requests_++;
  int32_t match = -1;
  auto& cache   = cache_[parms.rank_];
  for (int32_t idx = 0; idx < MAX_PLANS; ++idx) {
    auto& entry = cache[idx];
    if (nullptr == entry.plan) break;
    if (entry.parms == parms) {
      match = idx;
      cache_hits_++;
      break;
    }
  }

  float hit_rate = (float)cache_hits_ / cache_requests_;

  cufftPlan* result{nullptr};
  // If there's no match, we create a new plan
  if (-1 == match) {
    log_cudalibs.debug() << "[cufftPlanCache] no match found for " << parms.to_string()
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
                             << entry.parms.to_string() << " (type: " << type_ << ")";
        CHECK_CUFFT(cufftDestroy(entry.plan->handle));
        plan_index = idx;
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

    entry.parms = parms;
    result      = entry.plan.get();

    CHECK_CUFFT(cufftCreate(&result->handle));
    CHECK_CUFFT(cufftSetAutoAllocation(result->handle, 0 /*we'll do the allocation*/));

    CHECK_CUFFT(cufftMakePlanMany64(result->handle,
                                    entry.parms.rank_,
                                    entry.parms.n_,
                                    entry.parms.inembed_[0] != 0 ? entry.parms.inembed_ : nullptr,
                                    entry.parms.istride_,
                                    entry.parms.idist_,
                                    entry.parms.onembed_[0] != 0 ? entry.parms.onembed_ : nullptr,
                                    entry.parms.ostride_,
                                    entry.parms.odist_,
                                    type_,
                                    entry.parms.batch_,
                                    &result->workarea_size));
  }
  // Otherwise, we return the cached plan and adjust the LRU count
  else {
    log_cudalibs.debug() << "[cufftPlanCache] found match for " << parms.to_string()
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

cufftContext CUDALibraries::get_cufft_plan(cufftType type, const cufftPlanParms& parms)
{
  auto finder = plan_caches_.find(type);
  cufftPlanCache* cache{nullptr};

  if (plan_caches_.end() == finder) {
    cache              = new cufftPlanCache(type);
    plan_caches_[type] = cache;
  } else
    cache = finder->second;
  return cufftContext(cache->get_cufft_plan(parms));
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

cufftContext get_cufft_plan(cufftType type, const cufftPlanParms& parms)
{
  const auto proc = legate::Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cufft_plan(type, parms);
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
