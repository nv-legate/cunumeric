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

#include <map>
#include <mutex>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>

static void printtid(int op)
{
  pid_t tid;
  tid = syscall(SYS_gettid);
  ::fprintf(stderr, "[INFO-BITGENERATOR] : op = %d -- tid = %d -- pid = %d\n", op, tid, getpid());
}

#include "cunumeric/random/bitgenerator.h"
#include "cunumeric/random/bitgenerator_template.inl"
#include "cunumeric/random/bitgenerator_util.h"

#include "cunumeric/cuda_help.h"
#include "cunumeric/random/curand_help.h"

template <typename... args_t>
static void debug_trace_func(const char* filename, int line, const char* fmt, args_t... args)
{
  char format[1024];
  ::snprintf(format, 1024, "[DEBUG-TRACE] : @ %%s : %%d -- %s\n", fmt);
  ::fprintf(stdout, format, filename, line, args...);
}

#define DEBUG_TRACE_LINE(filename, line, ...) debug_trace_func(filename, line, __VA_ARGS__)
#if 0  // 1 for ongoing developments with full debug tracing
#define DEBUG_TRACE(...) DEBUG_TRACE_LINE(__FILE__, __LINE__, __VA_ARGS__)
#else
#define DEBUG_TRACE(...) \
  {                      \
  }
#endif

namespace cunumeric {

using namespace Legion;
using namespace legate;

struct CURANDGenerator {
  static constexpr size_t DEFAULT_DEV_BUFFER_SIZE = 64 * 1024;  // TODO: optimize this
  curandGenerator_t gen;
  uint64_t seed;
  uint64_t offset;
  curandRngType type;
  bool supports_skipahead;
  size_t dev_buffer_size;  // in number of entries
  uint32_t* dev_buffer;    // buffer for intermediate results

  std::mutex lock;  // in case several threads would want to use the same generator...

  CURANDGenerator() { DEBUG_TRACE("CURANDGenerator::create"); }

  ~CURANDGenerator() { DEBUG_TRACE("CURANDGenerator::destroy"); }

  void skip_ahead(uint64_t count)
  {
    if (supports_skipahead) {
      // skip ahead
      DEBUG_TRACE("count = %lu - offset = %lu", count, offset);
      offset += count;
      CHECK_CURAND(::curandSetGeneratorOffset(gen, offset));
    } else {
      // actually generate numbers in the temporary buffer
      DEBUG_TRACE("count = %lu - offset = %lu", count, offset);
      uint64_t remain = count;
      while (remain > 0) {
        if (remain < dev_buffer_size) {
          CHECK_CURAND(::curandGenerate(gen, dev_buffer, (size_t)remain));
          offset += remain;
          break;
        } else {
          CHECK_CURAND(::curandGenerate(gen, dev_buffer, (size_t)dev_buffer_size));
          offset += dev_buffer_size;
        }
        remain -= dev_buffer_size;
      }
    }
  }

  void generate_raw(uint64_t count, uint32_t* out)
  {
    CHECK_CURAND(::curandGenerate(gen, out, count));
    offset += count;
  }
};

struct generate_fn {
  template <int32_t DIM>
  size_t operator()(CURANDGenerator& gen,
                    legate::Store& output,
                    const DomainPoint& strides,
                    uint64_t totalcount)
  {
    auto rect       = output.shape<DIM>();
    uint64_t volume = rect.volume();

    uint64_t baseoffset = 0;
    for (size_t k = 0; k < DIM; ++k) baseoffset += rect.lo[k] * strides[k];

    assert(baseoffset + volume <= totalcount);

    uint64_t initialoffset = gen.offset;

    if (volume > 0) {
      auto out = output.write_accessor<uint32_t, DIM>(rect);

      if (!out.accessor.is_dense_row_major(rect))
        ::fprintf(stderr, "[ERROR] : accessor is not dense row major\n");
      assert(out.accessor.is_dense_row_major(rect));

      uint32_t* p = out.ptr(rect);

      if (baseoffset != 0) gen.skip_ahead(baseoffset);
      gen.generate_raw(volume, p);
    }

    // TODO: check if this is needed as setoffset is to be called on next call
    if (gen.offset != initialoffset + totalcount)
      gen.skip_ahead(initialoffset + totalcount - gen.offset);

    return totalcount;

#if 0
    ::fprintf(stdout, "[DEBUG] : total size = %llu\n", totalsize);

    auto domain = output.domain();
    ::fprintf(stdout, "[DEBUG] : domain = \n");
    for (int k = 0 ; k < DIM ; ++k)
      ::fprintf(stdout, "[DEBUG] : \t[%d] - [%lld:%lld]\n", k, domain.lo()[k], domain.hi()[k]);

    auto rect = output.shape<DIM>();
    ::fprintf(stdout, "[DEBUG] : shape = \n");
    for (int k = 0 ; k < DIM ; ++k)
      ::fprintf(stdout, "[DEBUG] : \t[%d] - [%lld:%lld]\n", k, rect.lo[k], rect.hi[k]);

    Pitches<DIM - 1> pitches;
    pitches.flatten(rect);

    // if (volume == 0) return;

    size_t volume = domain.get_volume();
    if (volume == 0) return 0;

    auto out = output.write_accessor<uint32_t, DIM>(rect);

    if (!out.accessor.is_dense_row_major(rect))
      ::fprintf(stderr, "[ERROR] : accessor is not dense row major\n");
    assert(out.accessor.is_dense_row_major(rect));

    Point<DIM,size_t> strd (strides);
    ::fprintf(stdout, "[DEBUG] : strides = \n");
    for (int k = 0 ; k < DIM ; ++k)
      ::fprintf(stdout, "[DEBUG] : \t[%d] = %zu\n", k, strd[k]);

    uint32_t* p = out.ptr(rect);
    DEBUG_TRACE("p = %p", p);

    CHECK_CURAND(::curandGenerate(gen.gen, p, volume));

    return volume;
#endif
  }
};

struct generatormap {
  generatormap() {}
  ~generatormap()
  {
    std::lock_guard<std::mutex> guard(lock);
    if (m_generators.size() != 0) {
      ::fprintf(stderr, "[ERROR] : some generators have not been freed - LEAK !\n");
      // TODO: assert ?
    }
  }

  std::mutex lock;
  std::map<int, std::unique_ptr<CURANDGenerator>> m_generators;

  bool has(int32_t generatorID)
  {
    std::lock_guard<std::mutex> guard(lock);
    return m_generators.find(generatorID) != m_generators.end();
  }

  CURANDGenerator* get(int32_t generatorID)
  {
    std::lock_guard<std::mutex> guard(lock);
    if (m_generators.find(generatorID) == m_generators.end()) {
      ::fprintf(stderr,
                "[ERROR] : internal error : generator ID <%d> does not exist (destroy) !\n",
                generatorID);
      assert(false);
    }
    return m_generators[generatorID].get();
  }

  void create(int32_t generatorID, BitGeneratorType gentype)
  {
    auto stream = get_cached_stream();
    curandGenerator_t gen;
    CHECK_CURAND(::curandCreateGenerator(&gen, get_curandRngType(gentype)));
    CHECK_CURAND(::curandSetStream(gen, stream));
    CURANDGenerator* cugenptr = new CURANDGenerator();
    CURANDGenerator& cugen    = *cugenptr;
    cugen.gen                 = gen;
    cugen.offset              = 0;
    cugen.type                = get_curandRngType(gentype);
    cugen.supports_skipahead  = supportsSkipAhead(cugen.type);
    cugen.dev_buffer_size     = cugen.DEFAULT_DEV_BUFFER_SIZE;
// TODO: use realm allocator
#if (__CUDACC_VER_MAJOR__ > 11 || ((__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 2)))
    CHECK_CUDA(
      ::cudaMallocAsync(&(cugen.dev_buffer), cugen.dev_buffer_size * sizeof(uint32_t), stream));
#else
    CHECK_CUDA(::cudaMalloc(&(cugen.dev_buffer), cugen.dev_buffer_size * sizeof(uint32_t)));
#endif

    std::lock_guard<std::mutex> guard(lock);
    // safety check
    if (m_generators.find(generatorID) != m_generators.end()) {
      ::fprintf(
        stderr, "[ERROR] : internal error : generator ID <%d> already in use !\n", generatorID);
      assert(false);
    }
    m_generators[generatorID] = std::move(std::unique_ptr<CURANDGenerator>(cugenptr));
  }

  void destroy(int32_t generatorID)
  {
    std::unique_ptr<CURANDGenerator> cugenptr;
    // verify it existed, and otherwise remove it from list
    {
      std::lock_guard<std::mutex> guard(lock);
      if (m_generators.find(generatorID) == m_generators.end()) {
        ::fprintf(stderr,
                  "[ERROR] : internal error : generator ID <%d> does not exist (destroy) !\n",
                  generatorID);
        assert(false);
      }
      cugenptr = std::move(m_generators[generatorID]);
      m_generators.erase(generatorID);
    }
    // wait for rand jobs and clean-up resources
    std::lock_guard<std::mutex> guard(cugenptr->lock);
    auto stream = get_cached_stream();
// TODO: use realm allocator
#if (__CUDACC_VER_MAJOR__ > 11 || ((__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 2)))
    CHECK_CUDA(::cudaFreeAsync(cugenptr->dev_buffer, stream));
#else
    CHECK_CUDA(::cudaFree(cugenptr->dev_buffer));
#endif
    CHECK_CURAND(::curandDestroyGenerator(cugenptr->gen));
  }

  void set_seed(int32_t generatorID, uint64_t seed)
  {
    CURANDGenerator* genptr = get(generatorID);
    std::lock_guard<std::mutex> guard(genptr->lock);
    CHECK_CURAND(::curandSetPseudoRandomGeneratorSeed(genptr->gen, seed));
  }
};

template <>
struct BitGeneratorImplBody<VariantKind::GPU> {
  static std::mutex lock_generators;
  static std::map<Legion::Processor, std::unique_ptr<generatormap>> m_generators;

 private:
  static generatormap& getgenmap()
  {
    const auto proc = Legion::Processor::get_executing_processor();
    lock_generators.lock();
    if (m_generators.find(proc) == m_generators.end()) {
      m_generators[proc] = std::move(std::unique_ptr<generatormap>(new generatormap()));
    }
    generatormap* res = m_generators[proc].get();
    lock_generators.unlock();
    return *res;
  }

 public:
  void operator()(BitGeneratorOperation op,
                  int32_t generatorID,
                  uint64_t parameter,
                  const DomainPoint& strides,
                  std::vector<legate::Store>& output,
                  std::vector<legate::Store>& args)
  {
    const auto proc = Legion::Processor::get_executing_processor();
    printtid((int)op);
    switch (op) {
      case BitGeneratorOperation::CREATE: {
        generatormap& genmap = getgenmap();

        if (genmap.has(generatorID)) {
          ::fprintf(
            stderr, "[ERROR] : internal error : generator ID <%d> already in use !\n", generatorID);
          assert(false);
        }

        genmap.create(generatorID, (BitGeneratorType)parameter);

        DEBUG_TRACE("created generator %d", generatorID);
      } break;
      case BitGeneratorOperation::DESTROY: {
        generatormap& genmap = getgenmap();

        genmap.destroy(generatorID);

        DEBUG_TRACE("destroyed generator %d", generatorID);
      } break;
      case BitGeneratorOperation::SET_SEED: {
        generatormap& genmap = getgenmap();

        genmap.set_seed(generatorID, parameter);

        DEBUG_TRACE("set seed %llu for generator %d", parameter, generatorID);
      } break;
      case BitGeneratorOperation::RAND_RAW: {
        generatormap& genmap = getgenmap();

        CURANDGenerator* genptr = genmap.get(generatorID);

        std::lock_guard<std::mutex> guard(genptr->lock);

        if (output.size() == 0) {
          CURANDGenerator& cugen = *genptr;
          cugen.skip_ahead(parameter);
        } else {
          CURANDGenerator& cugen = *genptr;
          legate::Store& res     = output[0];
          dim_dispatch(res.dim(), generate_fn{}, cugen, res, strides, parameter);
        }
      } break;
      default: {
        ::fprintf(stderr, "[ERROR] : unknown BitGenerator operation");
        assert(false);
      }
    }
  }
};

std::map<Legion::Processor, std::unique_ptr<generatormap>>
  BitGeneratorImplBody<VariantKind::GPU>::m_generators;
std::mutex BitGeneratorImplBody<VariantKind::GPU>::lock_generators;

/*static*/ void BitGeneratorTask::gpu_variant(legate::TaskContext& context)
{
  bitgenerator_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric