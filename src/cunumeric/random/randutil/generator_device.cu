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

#include "generator.cuh"
#include "generator_create.inl"

extern "C" curandStatus_t CURANDAPI randutilCreateGenerator(randutilGenerator_t* generator,
                                                            curandRngType_t rng_type,
                                                            uint64_t seed,
                                                            uint64_t generatorID,
                                                            uint32_t flags,
                                                            cudaStream_t stream)
{
  return inner_randutilCreateGenerator<randutilimpl::execlocation::DEVICE>(
    generator, rng_type, seed, generatorID, stream);
}

namespace randutilimpl {

// partially specialize dispatcher to enable DEVICE implementation generation
template <typename func_t, typename out_t>
struct dispatcher<randutilimpl::execlocation::DEVICE, func_t, out_t> {
  static curandStatus_t run(randutilimpl::basegenerator* gen, func_t func, size_t N, out_t* out)
  {
    return inner_dispatch_sample<randutilimpl::execlocation::DEVICE, func_t, out_t>(
      gen, func, N, out);
  }
};

}  // namespace randutilimpl

// explicit instantiations of distributions
#include "generator_integers.inl"
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, integers<int32_t>, int32_t>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, integers<int64_t>, int64_t>;
