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

#ifndef __NUMPY_PROJ_H__
#define __NUMPY_PROJ_H__

#include "numpy.h"

namespace legate {
namespace numpy {

// Interface for Legate projection functors
class NumPyProjectionFunctor : public Legion::ProjectionFunctor {
 public:
  NumPyProjectionFunctor(Legion::Runtime* runtime);

 public:
  using Legion::ProjectionFunctor::project;
  // Different projection methods for different branches
  virtual Legion::LogicalRegion project(Legion::LogicalPartition upper_bound,
                                        const Legion::DomainPoint& point,
                                        const Legion::Domain& launch_domain);

 public:
  virtual Legion::DomainPoint project_point(const Legion::DomainPoint& point,
                                            const Legion::Domain& launch_domain) const = 0;

 public:
  static void register_projection_functors(Legion::Runtime* runtime, Legion::ProjectionID offset);
  static NumPyProjectionFunctor* functors[NUMPY_PROJ_LAST];
  template <int DIM>
  static inline Legion::Rect<DIM> unpack_shape(const Legion::Task* task,
                                               legate::LegateDeserializer& derez)
  {
    const Legion::Point<DIM> shape = derez.template unpack_point<DIM>();
    const Legion::Rect<DIM> rect(Legion::Point<DIM>::ZEROES(), shape - Legion::Point<DIM>::ONES());
    // Unpack the projection functor ID
    const int functor_id = derez.unpack_32bit_int();
    // If the functor is less than zero than we know that this isn't valid
    // and we've got the actual rectangle that we need for this analysis
    if (functor_id < 0) return rect;
    // Otherwise intersect the shape with the chunk for our point
    const Legion::Point<DIM> chunk = derez.template unpack_point<DIM>();
    if (functor_id == 0) {
      // Default projection functor so we can do the easy thing
      const Legion::Point<DIM> local = task->index_point;
      const Legion::Point<DIM> lower = local * chunk;
      const Legion::Point<DIM> upper = lower + chunk - Legion::Point<DIM>::ONES();
      return rect.intersection(Legion::Rect<DIM>(lower, upper));
    } else {
      NumPyProjectionFunctor* functor = functors[functor_id];
      const Legion::Point<DIM> local =
        functor->project_point(task->index_point, task->index_domain);
      const Legion::Point<DIM> lower = local * chunk;
      const Legion::Point<DIM> upper = lower + chunk - Legion::Point<DIM>::ONES();
      return rect.intersection(Legion::Rect<DIM>(lower, upper));
    }
  }
};

class NumPyProjectionFunctor_2D_1D : public NumPyProjectionFunctor {
 public:
  NumPyProjectionFunctor_2D_1D(NumPyProjectionCode code, Legion::Runtime* runtime);

 public:
  virtual bool is_functional(void) const { return true; }
  virtual bool is_exclusive(void) const { return true; }
  virtual unsigned get_depth(void) const { return 0; }

 public:
  virtual Legion::DomainPoint project_point(const Legion::DomainPoint& point,
                                            const Legion::Domain& launch_domain) const;

 public:
  const NumPyProjectionCode code;
  const Legion::Transform<1, 2> transform;

 public:
  static Legion::Transform<1, 2> get_transform(NumPyProjectionCode code);
};

class NumPyProjectionFunctor_2D_2D : public NumPyProjectionFunctor {
 public:
  NumPyProjectionFunctor_2D_2D(NumPyProjectionCode code, Legion::Runtime* runtime);

 public:
  virtual bool is_functional(void) const { return true; }
  virtual bool is_exclusive(void) const { return true; }
  virtual unsigned get_depth(void) const { return 0; }

 public:
  virtual Legion::DomainPoint project_point(const Legion::DomainPoint& point,
                                            const Legion::Domain& launch_domain) const;

 public:
  const NumPyProjectionCode code;
  const Legion::Transform<2, 2> transform;

 public:
  static Legion::Transform<2, 2> get_transform(NumPyProjectionCode code);
};

class NumPyProjectionFunctor_1D_2D : public NumPyProjectionFunctor {
 public:
  NumPyProjectionFunctor_1D_2D(NumPyProjectionCode code, Legion::Runtime* runtime);

 public:
  virtual bool is_functional(void) const { return true; }
  virtual bool is_exclusive(void) const { return true; }
  virtual unsigned get_depth(void) const { return 0; }

 public:
  virtual Legion::DomainPoint project_point(const Legion::DomainPoint& point,
                                            const Legion::Domain& launch_domain) const;

 public:
  const NumPyProjectionCode code;
  const Legion::Transform<2, 1> transform;

 public:
  static Legion::Transform<2, 1> get_transform(NumPyProjectionCode code);
};

class NumPyProjectionFunctor_3D_2D : public NumPyProjectionFunctor {
 public:
  NumPyProjectionFunctor_3D_2D(NumPyProjectionCode code, Legion::Runtime* runtime);

 public:
  virtual bool is_functional(void) const { return true; }
  virtual bool is_exclusive(void) const { return true; }
  virtual unsigned get_depth(void) const { return 0; }

 public:
  virtual Legion::DomainPoint project_point(const Legion::DomainPoint& point,
                                            const Legion::Domain& launch_domain) const;

 public:
  const NumPyProjectionCode code;
  const Legion::Transform<2, 3> transform;

 public:
  static Legion::Transform<2, 3> get_transform(NumPyProjectionCode code);
};

class NumPyProjectionFunctor_3D_1D : public NumPyProjectionFunctor {
 public:
  NumPyProjectionFunctor_3D_1D(NumPyProjectionCode code, Legion::Runtime* runtime);

 public:
  virtual bool is_functional(void) const { return true; }
  virtual bool is_exclusive(void) const { return true; }
  virtual unsigned get_depth(void) const { return 0; }

 public:
  virtual Legion::DomainPoint project_point(const Legion::DomainPoint& point,
                                            const Legion::Domain& launch_domain) const;

 public:
  const NumPyProjectionCode code;
  const Legion::Transform<1, 3> transform;

 public:
  static Legion::Transform<1, 3> get_transform(NumPyProjectionCode code);
};

class NumPyProjectionFunctor_3D_3D : public NumPyProjectionFunctor {
 public:
  NumPyProjectionFunctor_3D_3D(NumPyProjectionCode code, Legion::Runtime* runtime);

 public:
  virtual bool is_functional(void) const { return true; }
  virtual bool is_exclusive(void) const { return true; }
  virtual unsigned get_depth(void) const { return 0; }

 public:
  virtual Legion::DomainPoint project_point(const Legion::DomainPoint& point,
                                            const Legion::Domain& launch_domain) const;

 public:
  const NumPyProjectionCode code;
  const Legion::Transform<3, 3> transform;

 public:
  static Legion::Transform<3, 3> get_transform(NumPyProjectionCode code);
};

class NumPyProjectionFunctor_2D_3D : public NumPyProjectionFunctor {
 public:
  NumPyProjectionFunctor_2D_3D(NumPyProjectionCode code, Legion::Runtime* runtime);

 public:
  virtual bool is_functional(void) const { return true; }
  virtual bool is_exclusive(void) const { return true; }
  virtual unsigned get_depth(void) const { return 0; }

 public:
  virtual Legion::DomainPoint project_point(const Legion::DomainPoint& point,
                                            const Legion::Domain& launch_domain) const;

 public:
  const NumPyProjectionCode code;
  const Legion::Transform<3, 2> transform;

 public:
  static Legion::Transform<3, 2> get_transform(NumPyProjectionCode code);
};

class NumPyProjectionFunctor_1D_3D : public NumPyProjectionFunctor {
 public:
  NumPyProjectionFunctor_1D_3D(NumPyProjectionCode code, Legion::Runtime* runtime);

 public:
  virtual bool is_functional(void) const { return true; }
  virtual bool is_exclusive(void) const { return true; }
  virtual unsigned get_depth(void) const { return 0; }

 public:
  virtual Legion::DomainPoint project_point(const Legion::DomainPoint& point,
                                            const Legion::Domain& launch_domain) const;

 public:
  const NumPyProjectionCode code;
  const Legion::Transform<3, 1> transform;

 public:
  static Legion::Transform<3, 1> get_transform(NumPyProjectionCode code);
};

}  // namespace numpy
}  // namespace legate

#endif  // __NUMPY_PROJ_H__
