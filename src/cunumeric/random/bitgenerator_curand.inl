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

#if 0  // for debugging
#define VERBOSE_DEBUGGING
#endif

#ifdef VERBOSE_DEBUGGING

static void printtid(int op)
{
  pid_t tid;
  tid = syscall(SYS_gettid);
  ::fprintf(stderr, "[INFO-BITGENERATOR] : op = %d -- tid = %d -- pid = %d\n", op, tid, getpid());
}

#else

static void printtid(int) {}

#endif

#include "cunumeric/random/bitgenerator.h"
#include "cunumeric/random/bitgenerator_template.inl"
#include "cunumeric/random/bitgenerator_util.h"

#include "cunumeric/random/curand_help.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

static Logger log_curand("cunumeric.random");

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

 protected:
  CURANDGenerator() { log_curand.debug() << "CURANDGenerator::create"; }

 public:
  virtual ~CURANDGenerator() { log_curand.debug() << "CURANDGenerator::destroy"; }

  void skip_ahead(uint64_t count)
  {
    if (supports_skipahead) {
      // skip ahead
      log_curand.debug() << "[skip-ahead] : count = " << count << " - offset = " << offset;
      offset += count;
      CHECK_CURAND(::curandSetGeneratorOffset(gen, offset));
    } else {
      // actually generate numbers in the temporary buffer
      log_curand.debug() << "[blank-generate] : count = " << count << " - offset = " << offset;
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
      log_curand.debug() << "\t[proc=" << proc.id << "] - shape[" << k << "][" << rect.lo[k] << ":"
                         << rect.hi[k] << "]";
    for (int k = 0; k < DIM; ++k)
      log_curand.debug() << "\t[proc=" << proc.id << "] - strides[" << k << "][" << strides[k]
                         << "]";

    uint64_t baseoffset = 0;
    for (size_t k = 0; k < DIM; ++k) baseoffset += rect.lo[k] * strides[k];

    log_curand.debug() << "[proc=" << proc.id << "] - base offset = " << baseoffset
                       << " - total count = " << totalcount;

    assert(baseoffset + volume <= totalcount);

    uint64_t initialoffset = gen.offset;

    if (volume > 0) {
      auto out = output.write_accessor<uint32_t, DIM>(rect);

      if (!out.accessor.is_dense_row_major(rect))
        log_curand.fatal() << "accessor is not dense row major - DIM = " << DIM;
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
struct generator_map {
  generator_map() {}
  ~generator_map()
  {
    std::lock_guard<std::mutex> guard(lock);
    if (m_generators.size() != 0) {
      log_curand.error() << "some generators have not been freed - LEAK !";
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
      log_curand.fatal() << "internal error : generator ID <" << generatorID
                         << "> does not exist (get) !";
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
      log_curand.fatal() << "internal error : generator ID <" << generatorID
                         << "> already in use !";
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
        log_curand.fatal() << "internal error : generator ID <" << generatorID
                           << "> does not exist (destroy) !";
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
  using generator_map_t = generator_map<kind>;

  static std::mutex lock_generators;
  static std::map<Legion::Processor, std::unique_ptr<generator_map_t>> m_generators;

 private:
  static generator_map_t& get_generator_map()
  {
    const auto proc = Legion::Processor::get_executing_processor();
    std::lock_guard<std::mutex> guard(lock_generators);
    if (m_generators.find(proc) == m_generators.end()) {
      m_generators[proc] = std::make_unique<generator_map_t>();
    }
    generator_map_t* res = m_generators[proc].get();
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
    printtid((int)op);
    generator_map_t& genmap = get_generator_map();
    switch (op) {
      case BitGeneratorOperation::CREATE: {
        genmap.create(generatorID, static_cast<BitGeneratorType>(parameter));

        log_curand.debug() << "created generator " << generatorID;
        break;
      }
      case BitGeneratorOperation::DESTROY: {
        genmap.destroy(generatorID);

        log_curand.debug() << "destroyed generator " << generatorID;
        break;
      }
      case BitGeneratorOperation::SET_SEED: {
        genmap.set_seed(generatorID, parameter);

        log_curand.debug() << "set seed " << parameter << " for generator " << generatorID;
        break;
      }
      case BitGeneratorOperation::RAND_RAW: {
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
          CURANDGenerator& cugen = *genptr;
          if (output.size() == 0) {
            cugen.skip_ahead(parameter);
          } else {
            legate::Store& res = output[0];
            dim_dispatch(res.dim(), generate_fn{}, cugen, res, strides, parameter);
          }
        }
        break;
      }
      default: {
        log_curand.fatal() << "unknown BitGenerator operation";
        assert(false);
      }
    }
  }
};

}  // namespace cunumeric