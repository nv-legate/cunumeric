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

UntypedPoint::~UntypedPoint() { destroy(); }

UntypedPoint::UntypedPoint(const UntypedPoint &other)
{
  destroy();
  copy(other);
}

UntypedPoint &UntypedPoint::operator=(const UntypedPoint &other)
{
  destroy();
  copy(other);
  return *this;
}

UntypedPoint::UntypedPoint(UntypedPoint &&other) noexcept
{
  destroy();
  move(std::forward<UntypedPoint>(other));
}

UntypedPoint &UntypedPoint::operator=(UntypedPoint &&other) noexcept
{
  destroy();
  move(std::forward<UntypedPoint>(other));
  return *this;
}

void UntypedPoint::copy(const UntypedPoint &other)
{
  if (exists()) {
    N_     = other.N_;
    point_ = dim_dispatch(N_, copy_fn{}, other.point_);
  }
}

void UntypedPoint::move(UntypedPoint &&other)
{
  N_     = other.N_;
  point_ = other.point_;

  other.N_     = -1;
  other.point_ = nullptr;
}

void UntypedPoint::destroy()
{
  if (exists()) {
    dim_dispatch(N_, destroy_fn{}, point_);
    N_ = -1;
  }
}

struct point_to_ostream_fn {
  template <int32_t N>
  void operator()(std::ostream &os, const UntypedPoint &point)
  {
    os << point.to_point<N>();
  }
};

std::ostream &operator<<(std::ostream &os, const UntypedPoint &point)
{
  dim_dispatch(point.dim(), point_to_ostream_fn{}, os, point);
  return os;
}

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

struct shape_to_ostream_fn {
  template <int32_t N>
  void operator()(std::ostream &os, const Shape &shape)
  {
    os << shape.to_rect<N>();
  }
};

std::ostream &operator<<(std::ostream &os, const Shape &shape)
{
  dim_dispatch(shape.dim(), shape_to_ostream_fn{}, os, shape);
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

RegionField::RegionField(int32_t dim, int32_t redop_id, const PhysicalRegion &pr, FieldID fid)
  : dim_(dim), redop_id_(redop_id), pr_(pr), fid_(fid)
{
}

RegionField::RegionField(
  int32_t dim, int32_t redop_id, const PhysicalRegion &pr, FieldID fid, Transform &&transform)
  : dim_(dim),
    redop_id_(redop_id),
    pr_(pr),
    fid_(fid),
    transform_(std::forward<Transform>(transform))
{
}

RegionField::RegionField(RegionField &&other) noexcept
  : dim_(other.dim_),
    redop_id_(other.redop_id_),
    pr_(other.pr_),
    fid_(other.fid_),
    transform_(std::forward<Transform>(other.transform_))
{
}

RegionField &RegionField::operator=(RegionField &&other) noexcept
{
  dim_       = other.dim_;
  redop_id_  = other.redop_id_;
  pr_        = other.pr_;
  fid_       = other.fid_;
  transform_ = std::move(other.transform_);
  return *this;
}

Array::Array(int32_t dim, LegateTypeCode code, Shape &&shape, Future future)
  : is_future_(true), dim_(dim), code_(code), shape_(std::forward<Shape>(shape)), future_(future)
{
}

Array::Array(int32_t dim, LegateTypeCode code, Shape &&shape, RegionField &&region_field)
  : is_future_(false),
    dim_(dim),
    code_(code),
    shape_(std::forward<Shape>(shape)),
    region_field_(std::forward<RegionField>(region_field))
{
}

Array &Array::operator=(Array &&other) noexcept
{
  is_future_ = other.is_future_;
  dim_       = other.dim_;
  code_      = other.code_;
  shape_     = std::move(other.shape_);
  if (is_future_)
    future_ = other.future_;
  else
    region_field_ = std::move(other.region_field_);
  return *this;
}

UntypedScalar Array::scalar() const { return future_.get_result<UntypedScalar>(); }

}  // namespace numpy
}  // namespace legate
