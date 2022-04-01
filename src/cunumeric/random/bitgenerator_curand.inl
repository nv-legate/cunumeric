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

#pragma once

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <mutex>

static void printtid(int op)
{
  pid_t tid;
  tid = syscall(SYS_gettid);
  ::fprintf(stderr, "[INFO-BITGENERATOR] : op = %d -- tid = %d -- pid = %d\n", op, tid, getpid());
}

#include "cunumeric/random/bitgenerator.h"
#include "cunumeric/random/bitgenerator_template.inl"
#include "cunumeric/random/bitgenerator_util.h"

// #include "cunumeric/cuda_help.h"
#include "cunumeric/random/curand_help.h"

template <typename... args_t>
static void debug_trace_func(const char* filename, int line, const char* fmt, args_t... args)
{
  char format[1024];
  ::snprintf(format, 1024, "[DEBUG-TRACE] : @ %%s : %%d -- %s\n", fmt);
  ::fprintf(stdout, format, filename, line, args...);
}

#define DEBUG_TRACE_LINE(filename, line, ...) debug_trace_func(filename, line, __VA_ARGS__)
#if 1  // 1 for ongoing developments with full debug tracing
#define DEBUG_TRACE(...) DEBUG_TRACE_LINE(__FILE__, __LINE__, __VA_ARGS__)
#else
#define DEBUG_TRACE(...) \
  {                      \
  }
#endif

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind kind>
struct CURANDGeneratorBuilder;

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
    const auto proc = Legion::Processor::get_executing_processor();

    auto rect       = output.shape<DIM>();
    uint64_t volume = rect.volume();

    for (int k = 0; k < DIM; ++k)
      DEBUG_TRACE("\t[proc=%llu] - shape[%d][%zu:%zu]", proc.id, k, rect.lo[k], rect.hi[k]);
    for (int k = 0; k < DIM; ++k)
      DEBUG_TRACE("\t[proc=%llu] - strides[%d][%zu]", proc.id, k, strides[k]);

    uint64_t baseoffset = 0;
    for (size_t k = 0; k < DIM; ++k) baseoffset += rect.lo[k] * strides[k];

    DEBUG_TRACE(
      "[proc=%llu] - base offset = %llu - total count = %llu", proc.id, baseoffset, totalcount);

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
  }
};

template <VariantKind kind>
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
    CURANDGenerator* cugenptr = CURANDGeneratorBuilder<kind>::build(gentype);

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

    CURANDGeneratorBuilder<kind>::destroy(cugenptr.get());
  }

  void set_seed(int32_t generatorID, uint64_t seed)
  {
    CURANDGenerator* genptr = get(generatorID);
    std::lock_guard<std::mutex> guard(genptr->lock);
    CHECK_CURAND(::curandSetPseudoRandomGeneratorSeed(genptr->gen, seed));
  }
};

template <VariantKind kind>
struct BitGeneratorImplBody {
  using generatormap_t = generatormap<kind>;

  static std::mutex lock_generators;
  static std::map<Legion::Processor, std::unique_ptr<generatormap_t>> m_generators;

 private:
  static generatormap_t& getgenmap()
  {
    const auto proc = Legion::Processor::get_executing_processor();
    lock_generators.lock();
    if (m_generators.find(proc) == m_generators.end()) {
      m_generators[proc] = std::move(std::unique_ptr<generatormap_t>(new generatormap_t()));
    }
    generatormap_t* res = m_generators[proc].get();
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
        generatormap_t& genmap = getgenmap();

        if (genmap.has(generatorID)) {
          ::fprintf(
            stderr, "[ERROR] : internal error : generator ID <%d> already in use !\n", generatorID);
          assert(false);
        }

        genmap.create(generatorID, (BitGeneratorType)parameter);

        DEBUG_TRACE("created generator %d", generatorID);
      } break;
      case BitGeneratorOperation::DESTROY: {
        generatormap_t& genmap = getgenmap();

        genmap.destroy(generatorID);

        DEBUG_TRACE("destroyed generator %d", generatorID);
      } break;
      case BitGeneratorOperation::SET_SEED: {
        generatormap_t& genmap = getgenmap();

        genmap.set_seed(generatorID, parameter);

        DEBUG_TRACE("set seed %llu for generator %d", parameter, generatorID);
      } break;
      case BitGeneratorOperation::RAND_RAW: {
        generatormap_t& genmap = getgenmap();

        CURANDGenerator* genptr = genmap.get(generatorID);

        if (isThreadSafe<kind == VariantKind::GPU>(genptr->type)) {
          std::lock_guard<std::mutex> guard(genptr->lock);

          if (output.size() == 0) {
            CURANDGenerator& cugen = *genptr;
            cugen.skip_ahead(parameter);
          } else {
            CURANDGenerator& cugen = *genptr;
            legate::Store& res     = output[0];
            dim_dispatch(res.dim(), generate_fn{}, cugen, res, strides, parameter);
          }
        } else {
          std::lock_guard<std::mutex> guard(genmap.lock);

          if (output.size() == 0) {
            CURANDGenerator& cugen = *genptr;
            cugen.skip_ahead(parameter);
          } else {
            CURANDGenerator& cugen = *genptr;
            legate::Store& res     = output[0];
            dim_dispatch(res.dim(), generate_fn{}, cugen, res, strides, parameter);
          }
        }
      } break;
      default: {
        ::fprintf(stderr, "[ERROR] : unknown BitGenerator operation");
        assert(false);
      }
    }
  }
};

}  // namespace cunumeric