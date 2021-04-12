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

LogicalRegion NumPyProjectionFunctor::project(LogicalPartition upper_bound, const DomainPoint& point, const Domain& launch_domain) {
  const DomainPoint dp = project_point(point, launch_domain);
  if (runtime->has_logical_subregion_by_color(upper_bound, dp))
    return runtime->get_logical_subregion_by_color(upper_bound, dp);
  else
    return LogicalRegion::NO_REGION;
}

NumPyProjectionFunctor_2D_1D::NumPyProjectionFunctor_2D_1D(NumPyProjectionCode c, Runtime* rt)
    : NumPyProjectionFunctor(rt), code(c), transform(get_transform(c)) {}

/*static*/ Transform<1, 2> NumPyProjectionFunctor_2D_1D::get_transform(NumPyProjectionCode code) {
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
    default:
      assert(false);
  }
  return result;
}

DomainPoint NumPyProjectionFunctor_2D_1D::project_point(const DomainPoint& p, const Domain& launch_domain) const {
  const Point<1> point = transform * Point<2>(p);
  return DomainPoint(point);
}

NumPyProjectionFunctor_2D_2D::NumPyProjectionFunctor_2D_2D(NumPyProjectionCode c, Runtime* rt)
    : NumPyProjectionFunctor(rt), code(c), transform(get_transform(c)) {}

/*static*/ Transform<2, 2> NumPyProjectionFunctor_2D_2D::get_transform(NumPyProjectionCode code) {
  Transform<2, 2> result;
  switch (code) {
    case NUMPY_PROJ_2D_2D_X0: {
      // 1, 0
      // 0, 0
      result[0][0] = 1;
      result[0][1] = 0;
      result[1][0] = 0;
      result[1][1] = 0;
      break;
    }
    case NUMPY_PROJ_2D_2D_0X: {
      // 0, 0
      // 1, 0
      result[0][0] = 0;
      result[0][1] = 0;
      result[1][0] = 1;
      result[1][1] = 0;
      break;
    }
    case NUMPY_PROJ_2D_2D_0Y: {
      // 0, 0
      // 0, 1
      result[0][0] = 0;
      result[0][1] = 0;
      result[1][0] = 0;
      result[1][1] = 1;
      break;
    }
    case NUMPY_PROJ_2D_2D_Y0: {
      // 0, 1
      // 0, 0
      result[0][0] = 0;
      result[0][1] = 1;
      result[1][0] = 0;
      result[1][1] = 0;
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
    default:
      assert(false);
  }
  return result;
}

DomainPoint NumPyProjectionFunctor_2D_2D::project_point(const DomainPoint& p, const Domain& launch_domain) const {
  const Point<2> point = transform * Point<2>(p);
  return DomainPoint(point);
}

NumPyProjectionFunctor_1D_2D::NumPyProjectionFunctor_1D_2D(NumPyProjectionCode c, Runtime* rt)
    : NumPyProjectionFunctor(rt), code(c), transform(get_transform(c)) {}

/*static*/ Transform<2, 1> NumPyProjectionFunctor_1D_2D::get_transform(NumPyProjectionCode code) {
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
    default:
      assert(false);
  }
  return result;
}

DomainPoint NumPyProjectionFunctor_1D_2D::project_point(const DomainPoint& p, const Domain& launch_domain) const {
  const Point<2> point = transform * Point<1>(p);
  return DomainPoint(point);
}

NumPyProjectionFunctor_3D_2D::NumPyProjectionFunctor_3D_2D(NumPyProjectionCode c, Runtime* rt)
    : NumPyProjectionFunctor(rt), code(c), transform(get_transform(c)) {}

/*static*/ Transform<2, 3> NumPyProjectionFunctor_3D_2D::get_transform(NumPyProjectionCode code) {
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
    case NUMPY_PROJ_3D_2D_XB: {
      // 0, 1, 0
      // 0, 0, 0
      result[0][0] = 0;
      result[0][1] = 1;
      result[0][2] = 0;
      result[1][0] = 0;
      result[1][1] = 0;
      result[1][2] = 0;
      break;
    }
    case NUMPY_PROJ_3D_2D_BY: {
      // 0, 0, 0
      // 0, 0, 1
      result[0][0] = 0;
      result[0][1] = 0;
      result[0][2] = 0;
      result[1][0] = 0;
      result[1][1] = 0;
      result[1][2] = 1;
      break;
    }
    default:
      assert(false);
  }
  return result;
}

DomainPoint NumPyProjectionFunctor_3D_2D::project_point(const DomainPoint& p, const Domain& launch_domain) const {
  const Point<2> point = transform * Point<3>(p);
  return DomainPoint(point);
}

NumPyProjectionFunctor_3D_1D::NumPyProjectionFunctor_3D_1D(NumPyProjectionCode c, Runtime* rt)
    : NumPyProjectionFunctor(rt), code(c), transform(get_transform(c)) {}

/*static*/ Transform<1, 3> NumPyProjectionFunctor_3D_1D::get_transform(NumPyProjectionCode code) {
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
    default:
      assert(false);
  }
  return result;
}

DomainPoint NumPyProjectionFunctor_3D_1D::project_point(const DomainPoint& p, const Domain& launch_domain) const {
  const Point<1> point = transform * Point<3>(p);
  return DomainPoint(point);
}

NumPyProjectionFunctor_3D_3D::NumPyProjectionFunctor_3D_3D(NumPyProjectionCode c, Runtime* rt)
    : NumPyProjectionFunctor(rt), code(c), transform(get_transform(c)) {}

/*static*/ Transform<3, 3> NumPyProjectionFunctor_3D_3D::get_transform(NumPyProjectionCode code) {
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
    default:
      assert(false);
  }
  return result;
}

DomainPoint NumPyProjectionFunctor_3D_3D::project_point(const DomainPoint& p, const Domain& launch_domain) const {
  const Point<3> point = transform * Point<3>(p);
  return DomainPoint(point);
}

NumPyProjectionFunctor_2D_3D::NumPyProjectionFunctor_2D_3D(NumPyProjectionCode c, Runtime* rt)
    : NumPyProjectionFunctor(rt), code(c), transform(get_transform(c)) {}

/*static*/ Transform<3, 2> NumPyProjectionFunctor_2D_3D::get_transform(NumPyProjectionCode code) {
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
    default:
      assert(false);
  }
  return result;
}

DomainPoint NumPyProjectionFunctor_2D_3D::project_point(const DomainPoint& p, const Domain& launch_domain) const {
  const Point<3> point = transform * Point<2>(p);
  return DomainPoint(point);
}

NumPyProjectionFunctor_1D_3D::NumPyProjectionFunctor_1D_3D(NumPyProjectionCode c, Runtime* rt)
    : NumPyProjectionFunctor(rt), code(c), transform(get_transform(c)) {}

/*static*/ Transform<3, 1> NumPyProjectionFunctor_1D_3D::get_transform(NumPyProjectionCode code) {
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
    default:
      assert(false);
  }
  return result;
}

DomainPoint NumPyProjectionFunctor_1D_3D::project_point(const DomainPoint& p, const Domain& launch_domain) const {
  const Point<3> point = transform * Point<1>(p);
  return DomainPoint(point);
}

NumPyProjectionFunctor_ND_1D_C_ORDER::NumPyProjectionFunctor_ND_1D_C_ORDER(NumPyProjectionCode c, Runtime* rt)
    : NumPyProjectionFunctor(rt), code(c) {}

DomainPoint NumPyProjectionFunctor_ND_1D_C_ORDER::project_point(const DomainPoint& p, const Domain& launch_domain) const {
  auto    hi             = launch_domain.hi();
  auto    lo             = launch_domain.lo();
  coord_t index          = p[0] - lo[0];
  coord_t partial_volume = hi[0] - lo[0] + 1;
  for (int dim = 1; dim < p.get_dim(); ++dim) {
    index += p[dim] * partial_volume;
    partial_volume *= hi[dim] - lo[dim] + 1;
  }
  return index;
}

template<bool LEFT>
NumPyProjectionFunctor_GEMV<LEFT>::NumPyProjectionFunctor_GEMV(NumPyProjectionCode c, Runtime* rt)
    : NumPyProjectionFunctor(rt), code(c) {}

template<bool LEFT>
DomainPoint NumPyProjectionFunctor_GEMV<LEFT>::project_point(const DomainPoint& p, const Domain& launch_domain) const {
  const Rect<3> launch_rect = launch_domain;
  assert(launch_rect.lo[2] == launch_rect.hi[2]);
  const Point<3> point = p;
  if (LEFT) {
    const coord_t  N   = (launch_rect.hi[1] - launch_rect.lo[1]) + 1;
    const Point<1> out = point[0] * N + (point[1] + point[2]) % N;
    return DomainPoint(out);
  } else {
    const coord_t  N   = (launch_rect.hi[0] - launch_rect.lo[0]) + 1;
    const Point<1> out = point[1] * N + (point[0] + point[2]) % N;
    return DomainPoint(out);
  }
}

template<int DIM, int RADIX, int OFFSET>
NumPyProjectionFunctorRadix2D<DIM, RADIX, OFFSET>::NumPyProjectionFunctorRadix2D(NumPyProjectionCode c, Runtime* rt)
    : NumPyProjectionFunctor(rt), code(c) {
  assert(DIM >= 0);
  assert(DIM < 2);
  assert(OFFSET < RADIX);
}

template<int DIM, int RADIX, int OFFSET>
DomainPoint NumPyProjectionFunctorRadix2D<DIM, RADIX, OFFSET>::project_point(const DomainPoint& p,
                                                                             const Domain&      launch_domain) const {
  const Rect<2>  launch_rect = launch_domain;
  const Point<2> point       = p;
  Point<2>       out         = point;
  out[DIM]                   = point[DIM] * RADIX + OFFSET;
  assert(launch_rect.lo[DIM] == 0);
  assert(out[DIM] < ((launch_rect.hi[DIM] + 1) * RADIX));
  return DomainPoint(out);
}

template<int DIM, int RADIX, int OFFSET>
NumPyProjectionFunctorRadix3D<DIM, RADIX, OFFSET>::NumPyProjectionFunctorRadix3D(NumPyProjectionCode c, Runtime* rt)
    : NumPyProjectionFunctor(rt), code(c) {
  assert(DIM >= 0);
  assert(DIM < 3);
  assert(OFFSET < RADIX);
}

template<int DIM, int RADIX, int OFFSET>
DomainPoint NumPyProjectionFunctorRadix3D<DIM, RADIX, OFFSET>::project_point(const DomainPoint& p,
                                                                             const Domain&      launch_domain) const {
  const Rect<3>  launch_rect = launch_domain;
  const Point<3> point       = p;
  Point<3>       out         = point;
  out[DIM]                   = point[DIM] * RADIX + OFFSET;
  assert(launch_rect.lo[DIM] == 0);
  assert(out[DIM] < ((launch_rect.hi[DIM] + 1) * RADIX));
  return DomainPoint(out);
}

template<typename T>
static void register_functor(Runtime* runtime, ProjectionID offset, NumPyProjectionCode code) {
  T* functor                             = new T(code, runtime);
  NumPyProjectionFunctor::functors[code] = functor;
  runtime->register_projection_functor((ProjectionID)(offset + code), functor, true /*silence warnings*/);
}

/*static*/ void NumPyProjectionFunctor::register_projection_functors(Runtime* runtime, ProjectionID offset) {
  // 2D reduction
  register_functor<NumPyProjectionFunctor_2D_1D>(runtime, offset, NUMPY_PROJ_2D_1D_X);
  register_functor<NumPyProjectionFunctor_2D_1D>(runtime, offset, NUMPY_PROJ_2D_1D_Y);
  // 2D broadcast
  register_functor<NumPyProjectionFunctor_2D_2D>(runtime, offset, NUMPY_PROJ_2D_2D_X0);
  register_functor<NumPyProjectionFunctor_2D_2D>(runtime, offset, NUMPY_PROJ_2D_2D_0X);
  register_functor<NumPyProjectionFunctor_2D_2D>(runtime, offset, NUMPY_PROJ_2D_2D_0Y);
  register_functor<NumPyProjectionFunctor_2D_2D>(runtime, offset, NUMPY_PROJ_2D_2D_Y0);
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
  register_functor<NumPyProjectionFunctor_3D_2D>(runtime, offset, NUMPY_PROJ_3D_2D_XB);
  register_functor<NumPyProjectionFunctor_3D_2D>(runtime, offset, NUMPY_PROJ_3D_2D_BY);
  // 3D promotion
  register_functor<NumPyProjectionFunctor_2D_3D>(runtime, offset, NUMPY_PROJ_2D_3D_XY);
  register_functor<NumPyProjectionFunctor_2D_3D>(runtime, offset, NUMPY_PROJ_2D_3D_XZ);
  register_functor<NumPyProjectionFunctor_2D_3D>(runtime, offset, NUMPY_PROJ_2D_3D_YZ);
  register_functor<NumPyProjectionFunctor_1D_3D>(runtime, offset, NUMPY_PROJ_1D_3D_X);
  register_functor<NumPyProjectionFunctor_1D_3D>(runtime, offset, NUMPY_PROJ_1D_3D_Y);
  register_functor<NumPyProjectionFunctor_1D_3D>(runtime, offset, NUMPY_PROJ_1D_3D_Z);
  // GEMV projection
  // register_functor<NumPyProjectionFunctor_GEMV<true> >(PROJ_GEMV_LEFT);
  // register_functor<NumPyProjectionFunctor_GEMV<false> >(PROJ_GEMV_RIGHT);
  // Radix 2D
  register_functor<NumPyProjectionFunctorRadix2D<0, 4, 0>>(runtime, offset, NUMPY_PROJ_RADIX_2D_X_4_0);
  register_functor<NumPyProjectionFunctorRadix2D<0, 4, 1>>(runtime, offset, NUMPY_PROJ_RADIX_2D_X_4_1);
  register_functor<NumPyProjectionFunctorRadix2D<0, 4, 2>>(runtime, offset, NUMPY_PROJ_RADIX_2D_X_4_2);
  register_functor<NumPyProjectionFunctorRadix2D<0, 4, 3>>(runtime, offset, NUMPY_PROJ_RADIX_2D_X_4_3);
  register_functor<NumPyProjectionFunctorRadix2D<1, 4, 0>>(runtime, offset, NUMPY_PROJ_RADIX_2D_Y_4_0);
  register_functor<NumPyProjectionFunctorRadix2D<1, 4, 1>>(runtime, offset, NUMPY_PROJ_RADIX_2D_Y_4_1);
  register_functor<NumPyProjectionFunctorRadix2D<1, 4, 2>>(runtime, offset, NUMPY_PROJ_RADIX_2D_Y_4_2);
  register_functor<NumPyProjectionFunctorRadix2D<1, 4, 3>>(runtime, offset, NUMPY_PROJ_RADIX_2D_Y_4_3);
  // Radix 3D
  register_functor<NumPyProjectionFunctorRadix3D<0, 4, 0>>(runtime, offset, NUMPY_PROJ_RADIX_3D_X_4_0);
  register_functor<NumPyProjectionFunctorRadix3D<0, 4, 1>>(runtime, offset, NUMPY_PROJ_RADIX_3D_X_4_1);
  register_functor<NumPyProjectionFunctorRadix3D<0, 4, 2>>(runtime, offset, NUMPY_PROJ_RADIX_3D_X_4_2);
  register_functor<NumPyProjectionFunctorRadix3D<0, 4, 3>>(runtime, offset, NUMPY_PROJ_RADIX_3D_X_4_3);
  register_functor<NumPyProjectionFunctorRadix3D<1, 4, 0>>(runtime, offset, NUMPY_PROJ_RADIX_3D_Y_4_0);
  register_functor<NumPyProjectionFunctorRadix3D<1, 4, 1>>(runtime, offset, NUMPY_PROJ_RADIX_3D_Y_4_1);
  register_functor<NumPyProjectionFunctorRadix3D<1, 4, 2>>(runtime, offset, NUMPY_PROJ_RADIX_3D_Y_4_2);
  register_functor<NumPyProjectionFunctorRadix3D<1, 4, 3>>(runtime, offset, NUMPY_PROJ_RADIX_3D_Y_4_3);
  register_functor<NumPyProjectionFunctorRadix3D<2, 4, 0>>(runtime, offset, NUMPY_PROJ_RADIX_3D_Z_4_0);
  register_functor<NumPyProjectionFunctorRadix3D<2, 4, 1>>(runtime, offset, NUMPY_PROJ_RADIX_3D_Z_4_1);
  register_functor<NumPyProjectionFunctorRadix3D<2, 4, 2>>(runtime, offset, NUMPY_PROJ_RADIX_3D_Z_4_2);
  register_functor<NumPyProjectionFunctorRadix3D<2, 4, 3>>(runtime, offset, NUMPY_PROJ_RADIX_3D_Z_4_3);
  // Flattening
  register_functor<NumPyProjectionFunctor_ND_1D_C_ORDER>(runtime, offset, NUMPY_PROJ_ND_1D_C_ORDER);
}

}    // namespace numpy
}    // namespace legate
