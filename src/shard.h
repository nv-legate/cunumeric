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

#ifndef __NUMPY_SHARD_H__
#define __NUMPY_SHARD_H__

#include "numpy.h"

namespace legate {
namespace numpy {

// Interface for Legate sharding functors
class NumPyShardingFunctor : public Legion::ShardingFunctor {
 public:
  NumPyShardingFunctor(NumPyShardingCode code);

 public:
  virtual Legion::ShardID shard(const Legion::DomainPoint& point,
                                const Legion::Domain& launch_space,
                                const size_t total_shards) = 0;
  // Compute the local index of the point for this shard
  // Used by the legate mapper to figure out which processor to use
  virtual unsigned localize(const Legion::DomainPoint& point,
                            const Legion::Domain& launch_space,
                            const size_t total_shards,
                            const Legion::ShardID local_shard) = 0;

 public:
  const NumPyShardingCode code;

 public:
  static void register_sharding_functors(Legion::Runtime* runtime, Legion::ShardingID offset);
  static NumPyShardingFunctor* sharding_functors[NUMPY_SHARD_LAST];
};

class NumPyShardingFunctor_1D : public NumPyShardingFunctor {
 public:
  NumPyShardingFunctor_1D(NumPyShardingCode code);

 public:
  virtual Legion::ShardID shard(const Legion::DomainPoint& point,
                                const Legion::Domain& launch_space,
                                const size_t total_shards);
  virtual unsigned localize(const Legion::DomainPoint& point,
                            const Legion::Domain& launch_space,
                            const size_t total_shards,
                            const Legion::ShardID local_shard);
};

class NumPyShardingFunctor_2D : public NumPyShardingFunctor {
 public:
  NumPyShardingFunctor_2D(NumPyShardingCode code);

 public:
  virtual Legion::ShardID shard(const Legion::DomainPoint& point,
                                const Legion::Domain& launch_space,
                                const size_t total_shards);
  virtual unsigned localize(const Legion::DomainPoint& point,
                            const Legion::Domain& launch_space,
                            const size_t total_shards,
                            const Legion::ShardID local_shard);
};

class NumPyShardingFunctor_2D_1D : public NumPyShardingFunctor {
 public:
  NumPyShardingFunctor_2D_1D(NumPyShardingCode code);

 public:
  virtual Legion::ShardID shard(const Legion::DomainPoint& point,
                                const Legion::Domain& launch_space,
                                const size_t total_shards);
  virtual unsigned localize(const Legion::DomainPoint& point,
                            const Legion::Domain& launch_space,
                            const size_t total_shards,
                            const Legion::ShardID local_shard);
};

class NumPyShardingFunctor_3D : public NumPyShardingFunctor {
 public:
  NumPyShardingFunctor_3D(NumPyShardingCode code);

 public:
  virtual Legion::ShardID shard(const Legion::DomainPoint& point,
                                const Legion::Domain& launch_space,
                                const size_t total_shards);
  virtual unsigned localize(const Legion::DomainPoint& point,
                            const Legion::Domain& launch_space,
                            const size_t total_shards,
                            const Legion::ShardID local_shard);
};

class NumPyShardingFunctor_3D_2D : public NumPyShardingFunctor {
 public:
  NumPyShardingFunctor_3D_2D(NumPyShardingCode code);

 public:
  virtual Legion::ShardID shard(const Legion::DomainPoint& point,
                                const Legion::Domain& launch_space,
                                const size_t total_shards);
  virtual unsigned localize(const Legion::DomainPoint& point,
                            const Legion::Domain& launch_space,
                            const size_t total_shards,
                            const Legion::ShardID local_shard);
};

// This is a transform sharding functor that will transform the shapes of incoming
// points into a different coordinate space and then use the base tiling functors
// to actually perform the sharding
template <int M, typename BASE>
class NumPyTransformShardingFunctor : public BASE {
 public:
  NumPyTransformShardingFunctor(NumPyShardingCode code, const long* transform, unsigned N);
  virtual ~NumPyTransformShardingFunctor(void);

 public:
  virtual Legion::ShardID shard(const Legion::DomainPoint& point,
                                const Legion::Domain& launch_space,
                                const size_t total_shards);
  virtual unsigned localize(const Legion::DomainPoint& point,
                            const Legion::Domain& launch_space,
                            const size_t total_shards,
                            const Legion::ShardID local_shard);

 protected:
  const unsigned N;
  long* const transform;
};

}  // namespace numpy
}  // namespace legate

#endif  // __NUMPY_SHARD_H__
