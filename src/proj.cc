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

#include "proj.h"

using namespace Legion;

namespace legate {
namespace numpy {

/*static*/ NumPyProjectionFunctor* NumPyProjectionFunctor::functors[NUMPY_PROJ_LAST];

NumPyProjectionFunctor::NumPyProjectionFunctor(Runtime* rt) : ProjectionFunctor(rt) {}

LogicalRegion NumPyProjectionFunctor::project(LogicalPartition upper_bound,
                                              const DomainPoint& point,
                                              const Domain& launch_domain)
{
  const DomainPoint dp = project_point(point, launch_domain);
  if (runtime->has_logical_subregion_by_color(upper_bound, dp))
    return runtime->get_logical_subregion_by_color(upper_bound, dp);
  else
    return LogicalRegion::NO_REGION;
}

NumPyProjectionFunctor_2D_1D::NumPyProjectionFunctor_2D_1D(NumPyProjectionCode c, Runtime* rt)
  : NumPyProjectionFunctor(rt), code(c), transform(get_transform(c))
{
}

/*static*/ Transform<1, 2> NumPyProjectionFunctor_2D_1D::get_transform(NumPyProjectionCode code)
{
  Transform<1, 2> result;
  switch (code) {
    case NUMPY_PROJ_2D_1D_X: {
      // 1, 0
      result[0][0] = 1;
      result[0][1] = 0;
      break;
    }
    case NUMPY_PROJ_2D_1D_Y: {
      // 0, 1
      result[0][0] = 0;
      result[0][1] = 1;
      break;
    }
    default: assert(false);
  }
  return result;
}

DomainPoint NumPyProjectionFunctor_2D_1D::project_point(const DomainPoint& p,
                                                        const Domain& launch_domain) const
{
  const Point<1> point = transform * Point<2>(p);
  return DomainPoint(point);
}

NumPyProjectionFunctor_2D_2D::NumPyProjectionFunctor_2D_2D(NumPyProjectionCode c, Runtime* rt)
  : NumPyProjectionFunctor(rt), code(c), transform(get_transform(c))
{
}

/*static*/ Transform<2, 2> NumPyProjectionFunctor_2D_2D::get_transform(NumPyProjectionCode code)
{
  Transform<2, 2> result;
  switch (code) {
    case NUMPY_PROJ_2D_2D_X: {
      // 1, 0
      // 0, 0
      result[0][0] = 1;
      result[0][1] = 0;
      result[1][0] = 0;
      result[1][1] = 0;
      break;
    }
    case NUMPY_PROJ_2D_2D_Y: {
      // 0, 0
      // 0, 1
      result[0][0] = 0;
      result[0][1] = 0;
      result[1][0] = 0;
      result[1][1] = 1;
      break;
    }
    case NUMPY_PROJ_2D_2D_YX: {
      // 0, 1
      // 1, 0
      result[0][0] = 0;
      result[0][1] = 1;
      result[1][0] = 1;
      result[1][1] = 0;
      break;
    }
    default: assert(false);
  }
  return result;
}

DomainPoint NumPyProjectionFunctor_2D_2D::project_point(const DomainPoint& p,
                                                        const Domain& launch_domain) const
{
  const Point<2> point = transform * Point<2>(p);
  return DomainPoint(point);
}

NumPyProjectionFunctor_1D_2D::NumPyProjectionFunctor_1D_2D(NumPyProjectionCode c, Runtime* rt)
  : NumPyProjectionFunctor(rt), code(c), transform(get_transform(c))
{
}

/*static*/ Transform<2, 1> NumPyProjectionFunctor_1D_2D::get_transform(NumPyProjectionCode code)
{
  Transform<2, 1> result;
  switch (code) {
    case NUMPY_PROJ_1D_2D_X: {
      // 1
      // 0
      result[0][0] = 1;
      result[1][0] = 0;
      break;
    }
    case NUMPY_PROJ_1D_2D_Y: {
      // 0
      // 1
      result[0][0] = 0;
      result[1][0] = 1;
      break;
    }
    default: assert(false);
  }
  return result;
}

DomainPoint NumPyProjectionFunctor_1D_2D::project_point(const DomainPoint& p,
                                                        const Domain& launch_domain) const
{
  const Point<2> point = transform * Point<1>(p);
  return DomainPoint(point);
}

NumPyProjectionFunctor_3D_2D::NumPyProjectionFunctor_3D_2D(NumPyProjectionCode c, Runtime* rt)
  : NumPyProjectionFunctor(rt), code(c), transform(get_transform(c))
{
}

/*static*/ Transform<2, 3> NumPyProjectionFunctor_3D_2D::get_transform(NumPyProjectionCode code)
{
  Transform<2, 3> result;
  switch (code) {
    case NUMPY_PROJ_3D_2D_XY: {
      // 1, 0, 0
      // 0, 1, 0
      result[0][0] = 1;
      result[0][1] = 0;
      result[0][2] = 0;
      result[1][0] = 0;
      result[1][1] = 1;
      result[1][2] = 0;
      break;
    }
    case NUMPY_PROJ_3D_2D_XZ: {
      // 1, 0, 0
      // 0, 0, 1
      result[0][0] = 1;
      result[0][1] = 0;
      result[0][2] = 0;
      result[1][0] = 0;
      result[1][1] = 0;
      result[1][2] = 1;
      break;
    }
    case NUMPY_PROJ_3D_2D_YZ: {
      // 0, 1, 0
      // 0, 0, 1
      result[0][0] = 0;
      result[0][1] = 1;
      result[0][2] = 0;
      result[1][0] = 0;
      result[1][1] = 0;
      result[1][2] = 1;
      break;
    }
    default: assert(false);
  }
  return result;
}

DomainPoint NumPyProjectionFunctor_3D_2D::project_point(const DomainPoint& p,
                                                        const Domain& launch_domain) const
{
  const Point<2> point = transform * Point<3>(p);
  return DomainPoint(point);
}

NumPyProjectionFunctor_3D_1D::NumPyProjectionFunctor_3D_1D(NumPyProjectionCode c, Runtime* rt)
  : NumPyProjectionFunctor(rt), code(c), transform(get_transform(c))
{
}

/*static*/ Transform<1, 3> NumPyProjectionFunctor_3D_1D::get_transform(NumPyProjectionCode code)
{
  Transform<1, 3> result;
  switch (code) {
    case NUMPY_PROJ_3D_1D_X: {
      // 1, 0, 0
      result[0][0] = 1;
      result[0][1] = 0;
      result[0][2] = 0;
      break;
    }
    case NUMPY_PROJ_3D_1D_Y: {
      // 0, 1, 0
      result[0][0] = 0;
      result[0][1] = 1;
      result[0][2] = 0;
      break;
    }
    case NUMPY_PROJ_3D_1D_Z: {
      // 0, 0, 1
      result[0][0] = 0;
      result[0][1] = 0;
      result[0][2] = 1;
      break;
    }
    default: assert(false);
  }
  return result;
}

DomainPoint NumPyProjectionFunctor_3D_1D::project_point(const DomainPoint& p,
                                                        const Domain& launch_domain) const
{
  const Point<1> point = transform * Point<3>(p);
  return DomainPoint(point);
}

NumPyProjectionFunctor_3D_3D::NumPyProjectionFunctor_3D_3D(NumPyProjectionCode c, Runtime* rt)
  : NumPyProjectionFunctor(rt), code(c), transform(get_transform(c))
{
}

/*static*/ Transform<3, 3> NumPyProjectionFunctor_3D_3D::get_transform(NumPyProjectionCode code)
{
  Transform<3, 3> result;
  switch (code) {
    case NUMPY_PROJ_3D_3D_XY: {
      // 1, 0, 0
      // 0, 1, 0
      // 0, 0, 0
      result[0][0] = 1;
      result[0][1] = 0;
      result[0][2] = 0;
      result[1][0] = 0;
      result[1][1] = 1;
      result[1][2] = 0;
      result[2][0] = 0;
      result[2][1] = 0;
      result[2][2] = 0;
      break;
    }
    case NUMPY_PROJ_3D_3D_XZ: {
      // 1, 0, 0
      // 0, 0, 0
      // 0, 0, 1
      result[0][0] = 1;
      result[0][1] = 0;
      result[0][2] = 0;
      result[1][0] = 0;
      result[1][1] = 0;
      result[1][2] = 0;
      result[2][0] = 0;
      result[2][1] = 0;
      result[2][2] = 1;
      break;
    }
    case NUMPY_PROJ_3D_3D_YZ: {
      // 0, 0, 0
      // 0, 1, 0
      // 0, 0, 1
      result[0][0] = 0;
      result[0][1] = 0;
      result[0][2] = 0;
      result[1][0] = 0;
      result[1][1] = 1;
      result[1][2] = 0;
      result[2][0] = 0;
      result[2][1] = 0;
      result[2][2] = 1;
      break;
    }
    case NUMPY_PROJ_3D_3D_X: {
      // 1, 0, 0
      // 0, 0, 0
      // 0, 0, 0
      result[0][0] = 1;
      result[0][1] = 0;
      result[0][2] = 0;
      result[1][0] = 0;
      result[1][1] = 0;
      result[1][2] = 0;
      result[2][0] = 0;
      result[2][1] = 0;
      result[2][2] = 0;
      break;
    }
    case NUMPY_PROJ_3D_3D_Y: {
      // 0, 0, 0
      // 0, 1, 0
      // 0, 0, 0
      result[0][0] = 0;
      result[0][1] = 0;
      result[0][2] = 0;
      result[1][0] = 0;
      result[1][1] = 1;
      result[1][2] = 0;
      result[2][0] = 0;
      result[2][1] = 0;
      result[2][2] = 0;
      break;
    }
    case NUMPY_PROJ_3D_3D_Z: {
      // 0, 0, 0
      // 0, 0, 0
      // 0, 0, 1
      result[0][0] = 0;
      result[0][1] = 0;
      result[0][2] = 0;
      result[1][0] = 0;
      result[1][1] = 0;
      result[1][2] = 0;
      result[2][0] = 0;
      result[2][1] = 0;
      result[2][2] = 1;
      break;
    }
    default: assert(false);
  }
  return result;
}

DomainPoint NumPyProjectionFunctor_3D_3D::project_point(const DomainPoint& p,
                                                        const Domain& launch_domain) const
{
  const Point<3> point = transform * Point<3>(p);
  return DomainPoint(point);
}

NumPyProjectionFunctor_2D_3D::NumPyProjectionFunctor_2D_3D(NumPyProjectionCode c, Runtime* rt)
  : NumPyProjectionFunctor(rt), code(c), transform(get_transform(c))
{
}

/*static*/ Transform<3, 2> NumPyProjectionFunctor_2D_3D::get_transform(NumPyProjectionCode code)
{
  Transform<3, 2> result;
  switch (code) {
    case NUMPY_PROJ_2D_3D_XY: {
      // 1, 0
      // 0, 1
      // 0, 0
      result[0][0] = 1;
      result[0][1] = 0;
      result[1][0] = 0;
      result[1][1] = 1;
      result[2][0] = 0;
      result[2][1] = 0;
      break;
    }
    case NUMPY_PROJ_2D_3D_XZ: {
      // 1, 0
      // 0, 0
      // 0, 1
      result[0][0] = 1;
      result[0][1] = 0;
      result[1][0] = 0;
      result[1][1] = 0;
      result[2][0] = 0;
      result[2][1] = 1;
      break;
    }
    case NUMPY_PROJ_2D_3D_YZ: {
      // 0, 0
      // 1, 0
      // 0, 1
      result[0][0] = 0;
      result[0][1] = 0;
      result[1][0] = 1;
      result[1][1] = 0;
      result[2][0] = 0;
      result[2][1] = 1;
      break;
    }
    default: assert(false);
  }
  return result;
}

DomainPoint NumPyProjectionFunctor_2D_3D::project_point(const DomainPoint& p,
                                                        const Domain& launch_domain) const
{
  const Point<3> point = transform * Point<2>(p);
  return DomainPoint(point);
}

NumPyProjectionFunctor_1D_3D::NumPyProjectionFunctor_1D_3D(NumPyProjectionCode c, Runtime* rt)
  : NumPyProjectionFunctor(rt), code(c), transform(get_transform(c))
{
}

/*static*/ Transform<3, 1> NumPyProjectionFunctor_1D_3D::get_transform(NumPyProjectionCode code)
{
  Transform<3, 1> result;
  switch (code) {
    case NUMPY_PROJ_1D_3D_X: {
      // 1, 0, 0
      result[0][0] = 1;
      result[0][1] = 0;
      result[0][2] = 0;
      break;
    }
    case NUMPY_PROJ_1D_3D_Y: {
      // 0, 1, 0
      result[0][0] = 0;
      result[0][1] = 1;
      result[0][2] = 0;
      break;
    }
    case NUMPY_PROJ_1D_3D_Z: {
      // 0, 0, 1
      result[0][0] = 0;
      result[0][1] = 0;
      result[0][2] = 1;
      break;
    }
    default: assert(false);
  }
  return result;
}

DomainPoint NumPyProjectionFunctor_1D_3D::project_point(const DomainPoint& p,
                                                        const Domain& launch_domain) const
{
  const Point<3> point = transform * Point<1>(p);
  return DomainPoint(point);
}

template <typename T>
static void register_functor(Runtime* runtime, ProjectionID offset, NumPyProjectionCode code)
{
  T* functor                             = new T(code, runtime);
  NumPyProjectionFunctor::functors[code] = functor;
  runtime->register_projection_functor(
    (ProjectionID)(offset + code), functor, true /*silence warnings*/);
}

/*static*/ void NumPyProjectionFunctor::register_projection_functors(Runtime* runtime,
                                                                     ProjectionID offset)
{
  // 2D reduction
  register_functor<NumPyProjectionFunctor_2D_1D>(runtime, offset, NUMPY_PROJ_2D_1D_X);
  register_functor<NumPyProjectionFunctor_2D_1D>(runtime, offset, NUMPY_PROJ_2D_1D_Y);
  // 2D broadcast
  register_functor<NumPyProjectionFunctor_2D_2D>(runtime, offset, NUMPY_PROJ_2D_2D_X);
  register_functor<NumPyProjectionFunctor_2D_2D>(runtime, offset, NUMPY_PROJ_2D_2D_Y);
  // 2D promotion
  register_functor<NumPyProjectionFunctor_1D_2D>(runtime, offset, NUMPY_PROJ_1D_2D_X);
  register_functor<NumPyProjectionFunctor_1D_2D>(runtime, offset, NUMPY_PROJ_1D_2D_Y);
  // 2D transpose
  register_functor<NumPyProjectionFunctor_2D_2D>(runtime, offset, NUMPY_PROJ_2D_2D_YX);
  // 3D reduction
  register_functor<NumPyProjectionFunctor_3D_2D>(runtime, offset, NUMPY_PROJ_3D_2D_XY);
  register_functor<NumPyProjectionFunctor_3D_2D>(runtime, offset, NUMPY_PROJ_3D_2D_XZ);
  register_functor<NumPyProjectionFunctor_3D_2D>(runtime, offset, NUMPY_PROJ_3D_2D_YZ);
  register_functor<NumPyProjectionFunctor_3D_1D>(runtime, offset, NUMPY_PROJ_3D_1D_X);
  register_functor<NumPyProjectionFunctor_3D_1D>(runtime, offset, NUMPY_PROJ_3D_1D_Y);
  register_functor<NumPyProjectionFunctor_3D_1D>(runtime, offset, NUMPY_PROJ_3D_1D_Z);
  // 3D broadcast
  register_functor<NumPyProjectionFunctor_3D_3D>(runtime, offset, NUMPY_PROJ_3D_3D_XY);
  register_functor<NumPyProjectionFunctor_3D_3D>(runtime, offset, NUMPY_PROJ_3D_3D_XZ);
  register_functor<NumPyProjectionFunctor_3D_3D>(runtime, offset, NUMPY_PROJ_3D_3D_YZ);
  register_functor<NumPyProjectionFunctor_3D_3D>(runtime, offset, NUMPY_PROJ_3D_3D_X);
  register_functor<NumPyProjectionFunctor_3D_3D>(runtime, offset, NUMPY_PROJ_3D_3D_Y);
  register_functor<NumPyProjectionFunctor_3D_3D>(runtime, offset, NUMPY_PROJ_3D_3D_Z);
  // 3D promotion
  register_functor<NumPyProjectionFunctor_2D_3D>(runtime, offset, NUMPY_PROJ_2D_3D_XY);
  register_functor<NumPyProjectionFunctor_2D_3D>(runtime, offset, NUMPY_PROJ_2D_3D_XZ);
  register_functor<NumPyProjectionFunctor_2D_3D>(runtime, offset, NUMPY_PROJ_2D_3D_YZ);
  register_functor<NumPyProjectionFunctor_1D_3D>(runtime, offset, NUMPY_PROJ_1D_3D_X);
  register_functor<NumPyProjectionFunctor_1D_3D>(runtime, offset, NUMPY_PROJ_1D_3D_Y);
  register_functor<NumPyProjectionFunctor_1D_3D>(runtime, offset, NUMPY_PROJ_1D_3D_Z);
}

}  // namespace numpy
}  // namespace legate
