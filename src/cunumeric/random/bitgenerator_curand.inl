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

#include "cunumeric/random/bitgenerator.h"
#include "cunumeric/random/bitgenerator_template.inl"
#include "cunumeric/random/bitgenerator_util.h"

#include "cunumeric/random/curand_help.h"
#include "cunumeric/random/curandex/curand_ex.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

static Logger log_curand("cunumeric.random");

template <VariantKind kind>
struct CURANDGeneratorBuilder;

struct CURANDGenerator {
  curandGeneratorEx_t gen_;
  uint64_t seed_;
  uint64_t generatorId_;
  curandRngType type_;

 protected:
  CURANDGenerator(BitGeneratorType gentype, uint64_t seed, uint64_t generatorId)
    : type_(get_curandRngType(gentype)), seed_(seed), generatorId_(generatorId)
  {
    log_curand.debug() << "CURANDGenerator::create";
  }

  CURANDGenerator(const CURANDGenerator&) = delete;

 public:
  virtual ~CURANDGenerator() { log_curand.debug() << "CURANDGenerator::destroy"; }

  void generate_raw(uint64_t count, uint32_t* out)
  {
    CHECK_CURAND(::curandGenerateRawUInt32Ex(gen_, out, count));
  }
  void generate_integer_64(uint64_t count, int64_t* out, int64_t low, int64_t high)
  {
    CHECK_CURAND(::curandGenerateIntegers64Ex(gen_, out, count, low, high));
  }
  void generate_integer_32(uint64_t count, int32_t* out, int32_t low, int32_t high)
  {
    CHECK_CURAND(::curandGenerateIntegers32Ex(gen_, out, count, low, high));
  }
};

struct generate_fn {
  template <int32_t DIM>
  size_t operator()(CURANDGenerator& gen, legate::Store& output)
  {
    auto rect       = output.shape<DIM>();
    uint64_t volume = rect.volume();

    const auto proc = Legion::Processor::get_executing_processor();
    log_curand.debug() << "proc=" << proc << " - shape = " << rect;

    if (volume > 0) {
      auto out = output.write_accessor<uint32_t, DIM>(rect);

      uint32_t* p = out.ptr(rect);

      gen.generate_raw(volume, p);
    }

    return volume;
  }
};

template <typename output_t>
struct integer_generator;
template <>
struct integer_generator<int64_t> {
  int64_t low_, high_;

  integer_generator(const std::vector<int64_t>& intparams,
                    const std::vector<float>& floatparams,
                    const std::vector<double>& doubleparams)
    : low_(intparams[0]), high_(intparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, int64_t* p) const
  {
    gen.generate_integer_64(count, p, low_, high_);
  }
};
template <>
struct integer_generator<int32_t> {
  int32_t low_, high_;

  integer_generator(const std::vector<int64_t>& intparams,
                    const std::vector<float>& floatparams,
                    const std::vector<double>& doubleparams)
    : low_((int32_t)intparams[0]), high_((int32_t)intparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, int32_t* p) const
  {
    gen.generate_integer_32(count, p, low_, high_);
  }
};

template <typename output_t, typename generator_t>
struct generate_distribution {
  const generator_t& generator_;

  generate_distribution(const generator_t& generator) : generator_(generator) {}

  template <int32_t DIM>
  size_t operator()(CURANDGenerator& gen, legate::Store& output)
  {
    auto rect       = output.shape<DIM>();
    uint64_t volume = rect.volume();

    const auto proc = Legion::Processor::get_executing_processor();
    log_curand.debug() << "proc=" << proc << " - shape = " << rect;

    if (volume > 0) {
      auto out = output.write_accessor<output_t, DIM>(rect);

      output_t* p = out.ptr(rect);

      generator_.generate(gen, volume, p);
    }

    return volume;
  }

  static void generate(legate::Store& res,
                       CURANDGenerator& cugen,
                       const std::vector<int64_t>& intparams,
                       const std::vector<float>& floatparams,
                       const std::vector<double>& doubleparams)
  {
    generator_t dist_gen(intparams, floatparams, doubleparams);
    generate_distribution<output_t, generator_t> generate_func(dist_gen);
    dim_dispatch(res.dim(), generate_func, cugen, res);
  }
};

template <VariantKind kind>
struct generator_map {
  generator_map() {}
  ~generator_map()
  {
    if (m_generators.size() != 0) {
      log_curand.debug() << "some generators have not been freed - cleaning-up !";
      // actually destroy
      for (auto kv = m_generators.begin(); kv != m_generators.end(); ++kv) {
        auto cugenptr = kv->second;
        CURANDGeneratorBuilder<kind>::destroy(cugenptr);
      }
      m_generators.clear();
    }
  }

  std::map<uint32_t, CURANDGenerator*> m_generators;

  bool has(uint32_t generatorID) { return m_generators.find(generatorID) != m_generators.end(); }

  CURANDGenerator* get(uint32_t generatorID)
  {
    if (m_generators.find(generatorID) == m_generators.end()) {
      log_curand.fatal() << "internal error : generator ID <" << generatorID
                         << "> does not exist (get) !";
      assert(false);
    }
    return m_generators[generatorID];
  }

  // called by the processor later using the generator
  void create(uint32_t generatorID, BitGeneratorType gentype, uint64_t seed, uint32_t flags)
  {
    const auto proc = Legion::Processor::get_executing_processor();
    CURANDGenerator* cugenptr =
      CURANDGeneratorBuilder<kind>::build(gentype, seed, (uint64_t)proc.id, flags);

    // safety check
    if (m_generators.find(generatorID) != m_generators.end()) {
      log_curand.fatal() << "internal error : generator ID <" << generatorID
                         << "> already in use !";
      assert(false);
    }
    m_generators[generatorID] = cugenptr;
  }

  void destroy(uint32_t generatorID)
  {
    CURANDGenerator* cugenptr;
    // verify it existed, and otherwise remove it from list
    {
      if (m_generators.find(generatorID) != m_generators.end()) {
        cugenptr = m_generators[generatorID];
        m_generators.erase(generatorID);
      } else
        // in some cases, destroy is forced, but processor never created the instance
        return;
    }

    CURANDGeneratorBuilder<kind>::destroy(cugenptr);
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
                  BitGeneratorType generatorType,  // to allow for lazy initialization,
                                                   // generatorType is always passed
                  uint64_t seed,   // to allow for lazy initialization, seed is always passed
                  uint32_t flags,  // for future use - ordering, etc.
                  BitGeneratorDistribution distribution,
                  const DomainPoint& strides,
                  std::vector<int64_t> intparams,
                  std::vector<float> floatparams,
                  std::vector<double> doubleparams,
                  std::vector<legate::Store>& output,
                  std::vector<legate::Store>& args)
  {
    generator_map_t& genmap = get_generator_map();
    // printtid((int)op);
    switch (op) {
      case BitGeneratorOperation::CREATE: {
        genmap.create(generatorID, generatorType, seed, flags);

        log_curand.debug() << "created generator " << generatorID;
        break;
      }
      case BitGeneratorOperation::DESTROY: {
        genmap.destroy(generatorID);

        log_curand.debug() << "destroyed generator " << generatorID;
        break;
      }
      case BitGeneratorOperation::RAND_RAW: {
        // allow for lazy initialization
        if (!genmap.has(generatorID)) genmap.create(generatorID, generatorType, seed, flags);
        // get the generator
        CURANDGenerator* genptr = genmap.get(generatorID);
        if (output.size() != 0) {
          legate::Store& res     = output[0];
          CURANDGenerator& cugen = *genptr;
          dim_dispatch(res.dim(), generate_fn{}, cugen, res);
        }
        break;
      }
      case BitGeneratorOperation::DISTRIBUTION: {
        // allow for lazy initialization
        if (!genmap.has(generatorID)) genmap.create(generatorID, generatorType, seed, flags);
        // get the generator
        CURANDGenerator* genptr = genmap.get(generatorID);
        if (output.size() != 0) {
          legate::Store& res     = output[0];
          CURANDGenerator& cugen = *genptr;
          switch (distribution) {
            case BitGeneratorDistribution::INTEGERS_32:
              generate_distribution<int32_t, integer_generator<int32_t>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::INTEGERS_64:
              generate_distribution<int64_t, integer_generator<int64_t>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            default: {
              log_curand.fatal() << "unknown Distribution";
              assert(false);
            }
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