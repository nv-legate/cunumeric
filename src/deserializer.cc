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

#include "deserializer.h"
#include "dispatch.h"
#include "proj.h"

namespace legate {
namespace numpy {

using namespace Legion;

Deserializer::Deserializer(const Task *task, const std::vector<PhysicalRegion> &regions)
  : task_{task},
    regions_{regions.data(), regions.size()},
    futures_{task->futures.data(), task->futures.size()},
    deserializer_{task->args, task->arglen},
    outputs_()
{
  auto runtime = Runtime::get_runtime();
  auto ctx     = Runtime::get_context();
  runtime->get_output_regions(ctx, outputs_);
}

void deserialize(Deserializer &ctx, __half &value) { value = ctx.deserializer_.unpack_half(); }

void deserialize(Deserializer &ctx, float &value) { value = ctx.deserializer_.unpack_float(); }

void deserialize(Deserializer &ctx, double &value) { value = ctx.deserializer_.unpack_double(); }

void deserialize(Deserializer &ctx, std::uint64_t &value)
{
  value = ctx.deserializer_.unpack_64bit_uint();
}

void deserialize(Deserializer &ctx, std::uint32_t &value)
{
  value = ctx.deserializer_.unpack_32bit_uint();
}

void deserialize(Deserializer &ctx, std::uint16_t &value)
{
  value = ctx.deserializer_.unpack_16bit_uint();
}

void deserialize(Deserializer &ctx, std::uint8_t &value)
{
  value = ctx.deserializer_.unpack_8bit_uint();
}

void deserialize(Deserializer &ctx, std::int64_t &value)
{
  value = ctx.deserializer_.unpack_64bit_int();
}

void deserialize(Deserializer &ctx, std::int32_t &value)
{
  value = ctx.deserializer_.unpack_32bit_int();
}

void deserialize(Deserializer &ctx, std::int16_t &value)
{
  value = ctx.deserializer_.unpack_64bit_int();
}

void deserialize(Deserializer &ctx, std::int8_t &value)
{
  value = ctx.deserializer_.unpack_8bit_int();
}

void deserialize(Deserializer &ctx, std::string &value)
{
  value = ctx.deserializer_.unpack_string();
}

void deserialize(Deserializer &ctx, bool &value) { value = ctx.deserializer_.unpack_bool(); }

void deserialize(Deserializer &ctx, LegateTypeCode &code)
{
  code = static_cast<LegateTypeCode>(ctx.deserializer_.unpack_32bit_int());
}

struct deserialize_shape_fn {
  template <int N>
  Shape operator()(const Task *task, LegateDeserializer &ctx)
  {
    return Shape(NumPyProjectionFunctor::unpack_shape<N>(task, ctx));
  }
};

void deserialize(Deserializer &ctx, Shape &value)
{
  auto dim = ctx.deserializer_.unpack_32bit_int();
  if (dim < 0) return;
  value = dim_dispatch(dim, deserialize_shape_fn{}, ctx.task_, ctx.deserializer_);
}

struct deserialize_transform_fn {
  template <int M, int N>
  Transform operator()(LegateDeserializer &ctx)
  {
    return Transform(ctx.unpack_transform<M, N>());
  }
};

void deserialize(Deserializer &ctx, Transform &value)
{
  auto M = ctx.deserializer_.unpack_32bit_int();
  if (M < 0) return;
  auto N = ctx.deserializer_.unpack_32bit_int();
  value  = double_dispatch(M, N, deserialize_transform_fn{}, ctx.deserializer_);
}

void deserialize(Deserializer &ctx, RegionField &value)
{
  auto dim  = ctx.deserializer_.unpack_32bit_int();
  auto code = ctx.deserializer_.unpack_dtype();

  auto idx = ctx.deserializer_.unpack_32bit_uint();
  auto &pr = ctx.regions_[idx];
  auto fid = ctx.deserializer_.unpack_32bit_int();

  Transform transform;
  deserialize(ctx, transform);

  value = RegionField(dim, code, pr, fid, std::move(transform));
}

}  // namespace numpy
}  // namespace legate
