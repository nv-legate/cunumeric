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

#include "core.h"
#include "dispatch.h"
#include "scalar.h"

namespace legate {
namespace numpy {

using namespace Legion;

Shape::~Shape() { destroy(); }

Shape::Shape(const Shape &other)
{
  destroy();
  copy(other);
}

Shape &Shape::operator=(const Shape &other)
{
  destroy();
  copy(other);
  return *this;
}

Shape::Shape(Shape &&other) noexcept
{
  destroy();
  move(std::forward<Shape>(other));
}

Shape &Shape::operator=(Shape &&other) noexcept
{
  destroy();
  move(std::forward<Shape>(other));
  return *this;
}

void Shape::copy(const Shape &other)
{
  if (exists()) {
    N_    = other.N_;
    rect_ = dim_dispatch(N_, copy_fn{}, other.rect_);
  }
}

void Shape::move(Shape &&other)
{
  N_    = other.N_;
  rect_ = other.rect_;

  other.N_    = -1;
  other.rect_ = nullptr;
}

void Shape::destroy()
{
  if (exists()) {
    dim_dispatch(N_, destroy_fn{}, rect_);
    N_ = -1;
  }
}

struct pipe_to_ostream_fn {
  template <int32_t N>
  void operator()(std::ostream &os, const Shape &shape)
  {
    os << shape.to_rect<N>();
  }
};

std::ostream &operator<<(std::ostream &os, const Shape &shape)
{
  dim_dispatch(shape.dim(), pipe_to_ostream_fn{}, os, shape);
  return os;
}

Transform::~Transform() { destroy(); }

Transform::Transform(const Transform &other)
{
  destroy();
  copy(other);
}

Transform &Transform::operator=(const Transform &other)
{
  destroy();
  copy(other);
  return *this;
}

Transform::Transform(Transform &&other) noexcept
{
  destroy();
  move(std::forward<Transform>(other));
}

Transform &Transform::operator=(Transform &&other) noexcept
{
  destroy();
  move(std::forward<Transform>(other));
  return *this;
}

void Transform::copy(const Transform &other)
{
  if (exists()) {
    M_         = other.M_;
    N_         = other.N_;
    transform_ = double_dispatch(M_, N_, copy_fn{}, other.transform_);
  }
}

void Transform::move(Transform &&other)
{
  transform_ = other.transform_;
  M_         = other.M_;
  N_         = other.N_;

  other.M_         = -1;
  other.N_         = -1;
  other.transform_ = nullptr;
}

void Transform::destroy()
{
  if (exists()) {
    double_dispatch(M_, N_, destroy_fn{}, transform_);
    M_ = -1;
    N_ = -1;
  }
}

RegionField::RegionField(int dim, LegateTypeCode code, const PhysicalRegion &pr, FieldID fid)
  : dim_(dim), code_(code), pr_(pr), fid_(fid)
{
}

RegionField::RegionField(
  int dim, LegateTypeCode code, const PhysicalRegion &pr, FieldID fid, Transform &&transform)
  : dim_(dim), code_(code), pr_(pr), fid_(fid), transform_(std::forward<Transform>(transform))
{
}

RegionField::RegionField(RegionField &&other) noexcept
  : dim_(other.dim_),
    code_(other.code_),
    pr_(other.pr_),
    fid_(other.fid_),
    transform_(std::forward<Transform>(other.transform_))
{
}

RegionField &RegionField::operator=(RegionField &&other) noexcept
{
  dim_       = other.dim_;
  code_      = other.code_;
  pr_        = other.pr_;
  fid_       = other.fid_;
  transform_ = std::move(other.transform_);
  return *this;
}

Array::Array(int32_t dim, LegateTypeCode code, Future future)
  : is_future_(true), dim_(dim), code_(code), future_(future)
{
}

Array::Array(int32_t dim, LegateTypeCode code, RegionField &&region_field)
  : is_future_(false),
    dim_(dim),
    code_(code),
    region_field_(std::forward<RegionField>(region_field))
{
}

Array &Array::operator=(Array &&other) noexcept
{
  is_future_ = other.is_future_;
  dim_       = other.dim_;
  code_      = other.code_;
  if (is_future_)
    future_ = other.future_;
  else
    region_field_ = std::move(other.region_field_);
  return *this;
}

UntypedScalar Array::scalar() const { return future_.get_result<UntypedScalar>(); }

}  // namespace numpy
}  // namespace legate
