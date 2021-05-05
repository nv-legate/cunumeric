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

#pragma once

#include "numpy.h"

namespace legate {
namespace numpy {

class Shape {
 public:
  Shape() noexcept {}
  ~Shape();

 public:
  template <int N>
  Shape(const Legion::Rect<N> &rect) : N_(N), rect_(new Legion::Rect<N>(rect))
  {
  }

 public:
  Shape(const Shape &other);
  Shape &operator=(const Shape &other);

 public:
  Shape(Shape &&other) noexcept;
  Shape &operator=(Shape &&other) noexcept;

 public:
  int dim() const noexcept { return N_; }
  bool exists() const noexcept { return nullptr != rect_; }

 public:
  template <int N>
  Legion::Rect<N> to_rect() const
  {
#ifdef LEGION_BOUNDS_CHECKS
    assert(N_ == N);
#endif
    return *static_cast<Legion::Rect<N> *>(rect_);
  }

 private:
  struct copy_fn {
    template <int N>
    void *operator()(void *rect)
    {
      return new Legion::Rect<N>(*static_cast<Legion::Rect<N> *>(rect));
    }
  };
  void copy(const Shape &other);
  void move(Shape &&other);
  struct destroy_fn {
    template <int N>
    void operator()(void *rect)
    {
      delete static_cast<Legion::Rect<N> *>(rect);
    }
  };
  void destroy();

 private:
  int N_{-1};
  void *rect_{nullptr};
};

class Transform {
 public:
  Transform() noexcept {}
  ~Transform();

 public:
  template <int M, int N>
  Transform(const Legion::AffineTransform<M, N> &transform)
    : M_(M), N_(N), transform_(new Legion::AffineTransform<M, N>(transform))
  {
  }

 public:
  Transform(const Transform &other);
  Transform &operator=(const Transform &other);

 public:
  Transform(Transform &&other) noexcept;
  Transform &operator=(Transform &&other) noexcept;

 public:
  std::pair<int, int> shape() const { return std::make_pair(M_, N_); }
  bool exists() const { return nullptr != transform_; }

 public:
  template <int M, int N>
  Legion::AffineTransform<M, N> to_affine_transform() const
  {
#ifdef LEGION_BOUNDS_CHECKS
    assert(M_ == M && N_ == N);
#endif
    return *static_cast<Legion::AffineTransform<M, N> *>(transform_);
  }

 private:
  struct copy_fn {
    template <int M, int N>
    void *operator()(void *transform)
    {
      using Transform = Legion::AffineTransform<M, N>;
      return new Transform(*static_cast<Transform *>(transform));
    }
  };
  void copy(const Transform &other);
  void move(Transform &&other);
  struct destroy_fn {
    template <int M, int N>
    void operator()(void *transform)
    {
      using Transform = Legion::AffineTransform<M, N>;
      delete static_cast<Transform *>(transform);
    }
  };
  void destroy();

 private:
  int M_{-1};
  int N_{-1};
  void *transform_{nullptr};
};

class RegionField {
 public:
  RegionField() {}
  RegionField(int dim, LegateTypeCode code, const Legion::PhysicalRegion &pr, Legion::FieldID fid);
  RegionField(int dim,
              LegateTypeCode code,
              const Legion::PhysicalRegion &pr,
              Legion::FieldID fid,
              Transform &&transform);

 public:
  RegionField(RegionField &&other) noexcept;
  RegionField &operator=(RegionField &&other) noexcept;

 private:
  RegionField(const RegionField &other) = delete;
  RegionField &operator=(const RegionField &other) = delete;

 public:
  int dim() const { return dim_; }
  LegateTypeCode code() const { return code_; }

 public:
  template <typename T, int N>
  struct read_trans_accesor_fn {
    template <int M>
    AccessorRO<T, N> operator()(const Legion::PhysicalRegion &pr,
                                Legion::FieldID fid,
                                const Transform &transform)
    {
      auto trans = transform.to_affine_transform<M, N>();
      return AccessorRO<T, N>(pr, fid, trans);
    }
  };
  template <typename T, int DIM>
  AccessorRO<T, DIM> read_accessor(void) const;

  template <typename T, int N>
  struct write_trans_accesor_fn {
    template <int M>
    AccessorWO<T, N> operator()(const Legion::PhysicalRegion &pr,
                                Legion::FieldID fid,
                                const Transform &transform)
    {
      auto trans = transform.to_affine_transform<M, N>();
      return AccessorWO<T, N>(pr, fid, trans);
    }
  };
  template <typename T, int DIM>
  AccessorWO<T, DIM> write_accessor(void) const;

 private:
  int dim_{-1};
  LegateTypeCode code_{MAX_TYPE_NUMBER};
  Legion::PhysicalRegion pr_{};
  Legion::FieldID fid_{-1U};
  Transform transform_{};

 private:
  bool readable_{false};
  bool writable_{false};
  bool reducible_{false};
};

template <typename T, int DIM>
AccessorRO<T, DIM> RegionField::read_accessor(void) const
{
  if (transform_.exists())
    return dim_dispatch(
      transform_.shape().first, read_trans_accesor_fn<T, DIM>{}, pr_, fid_, transform_);
  else {
#ifdef LEGION_BOUNDS_CHECKS
    assert(DIM == dim());
#endif
    return AccessorRO<T, DIM>(pr_, fid_);
  }
}

template <typename T, int DIM>
AccessorWO<T, DIM> RegionField::write_accessor(void) const
{
  if (transform_.exists())
    return dim_dispatch(
      transform_.shape().first, write_trans_accesor_fn<T, DIM>{}, pr_, fid_, transform_);
  else {
#ifdef LEGION_BOUNDS_CHECKS
    assert(DIM == dim());
#endif
    return AccessorWO<T, DIM>(pr_, fid_);
  }
}

}  // namespace numpy
}  // namespace legate
