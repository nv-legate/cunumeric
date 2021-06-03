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

class UntypedScalar;

class UntypedPoint {
 public:
  UntypedPoint() noexcept {}
  ~UntypedPoint();

 public:
  template <int32_t N>
  UntypedPoint(const Legion::Point<N> &point) : N_(N), point_(new Legion::Point<N>(point))
  {
  }

 public:
  UntypedPoint(const UntypedPoint &other);
  UntypedPoint &operator=(const UntypedPoint &other);

 public:
  UntypedPoint(UntypedPoint &&other) noexcept;
  UntypedPoint &operator=(UntypedPoint &&other) noexcept;

 public:
  int32_t dim() const noexcept { return N_; }
  bool exists() const noexcept { return nullptr != point_; }

 public:
  template <int32_t N>
  Legion::Point<N> to_point() const
  {
    assert(N_ == N);
    return *static_cast<Legion::Point<N> *>(point_);
  }

 private:
  struct copy_fn {
    template <int32_t N>
    void *operator()(void *point)
    {
      return new Legion::Point<N>(*static_cast<Legion::Point<N> *>(point));
    }
  };
  void copy(const UntypedPoint &other);
  void move(UntypedPoint &&other);
  struct destroy_fn {
    template <int32_t N>
    void operator()(void *point)
    {
      delete static_cast<Legion::Point<N> *>(point);
    }
  };
  void destroy();

 private:
  int32_t N_{-1};
  void *point_{nullptr};
};

std::ostream &operator<<(std::ostream &os, const UntypedPoint &point);

class Shape {
 public:
  Shape() noexcept {}
  ~Shape();

 public:
  template <int32_t N>
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
  int32_t dim() const noexcept { return N_; }
  bool exists() const noexcept { return nullptr != rect_; }

 public:
  template <int32_t N>
  Legion::Rect<N> to_rect() const
  {
    assert(N_ == N);
    return *static_cast<Legion::Rect<N> *>(rect_);
  }

 private:
  struct copy_fn {
    template <int32_t N>
    void *operator()(void *rect)
    {
      return new Legion::Rect<N>(*static_cast<Legion::Rect<N> *>(rect));
    }
  };
  void copy(const Shape &other);
  void move(Shape &&other);
  struct destroy_fn {
    template <int32_t N>
    void operator()(void *rect)
    {
      delete static_cast<Legion::Rect<N> *>(rect);
    }
  };
  void destroy();

 private:
  int32_t N_{-1};
  void *rect_{nullptr};
};

std::ostream &operator<<(std::ostream &os, const Shape &shape);

class Transform {
 public:
  Transform() noexcept {}
  ~Transform();

 public:
  template <int32_t M, int32_t N>
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
  std::pair<int32_t, int32_t> shape() const { return std::make_pair(M_, N_); }
  bool exists() const { return nullptr != transform_; }

 public:
  template <int32_t M, int32_t N>
  Legion::AffineTransform<M, N> to_affine_transform() const
  {
    assert(M_ == M && N_ == N);
    return *static_cast<Legion::AffineTransform<M, N> *>(transform_);
  }

 private:
  struct copy_fn {
    template <int32_t M, int32_t N>
    void *operator()(void *transform)
    {
      using Transform = Legion::AffineTransform<M, N>;
      return new Transform(*static_cast<Transform *>(transform));
    }
  };
  void copy(const Transform &other);
  void move(Transform &&other);
  struct destroy_fn {
    template <int32_t M, int32_t N>
    void operator()(void *transform)
    {
      using Transform = Legion::AffineTransform<M, N>;
      delete static_cast<Transform *>(transform);
    }
  };
  void destroy();

 private:
  int32_t M_{-1};
  int32_t N_{-1};
  void *transform_{nullptr};
};

class RegionField {
 public:
  RegionField() {}
  RegionField(int32_t dim, int32_t redop_id, const Legion::PhysicalRegion &pr, Legion::FieldID fid);
  RegionField(int32_t dim,
              int32_t redop_id,
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
  int32_t dim() const { return dim_; }

 private:
  template <typename T, int32_t N>
  struct read_trans_accesor_fn {
    template <int32_t M>
    AccessorRO<T, N> operator()(const Legion::PhysicalRegion &pr,
                                Legion::FieldID fid,
                                const Transform &transform)
    {
      auto trans = transform.to_affine_transform<M, N>();
      return AccessorRO<T, N>(pr, fid, trans);
    }
    template <int32_t M>
    AccessorRO<T, N> operator()(const Legion::PhysicalRegion &pr,
                                Legion::FieldID fid,
                                const Transform &transform,
                                const Legion::Rect<N> &bounds)
    {
      auto trans = transform.to_affine_transform<M, N>();
      return AccessorRO<T, N>(pr, fid, trans, bounds);
    }
  };

  template <typename T, int32_t N>
  struct write_trans_accesor_fn {
    template <int32_t M>
    AccessorWO<T, N> operator()(const Legion::PhysicalRegion &pr,
                                Legion::FieldID fid,
                                const Transform &transform)
    {
      auto trans = transform.to_affine_transform<M, N>();
      return AccessorWO<T, N>(pr, fid, trans);
    }
    template <int32_t M>
    AccessorWO<T, N> operator()(const Legion::PhysicalRegion &pr,
                                Legion::FieldID fid,
                                const Transform &transform,
                                const Legion::Rect<N> &bounds)
    {
      auto trans = transform.to_affine_transform<M, N>();
      return AccessorWO<T, N>(pr, fid, trans, bounds);
    }
  };

  template <typename T, int32_t N>
  struct read_write_trans_accesor_fn {
    template <int32_t M>
    AccessorRW<T, N> operator()(const Legion::PhysicalRegion &pr,
                                Legion::FieldID fid,
                                const Transform &transform)
    {
      auto trans = transform.to_affine_transform<M, N>();
      return AccessorRW<T, N>(pr, fid, trans);
    }
    template <int32_t M>
    AccessorRW<T, N> operator()(const Legion::PhysicalRegion &pr,
                                Legion::FieldID fid,
                                const Transform &transform,
                                const Legion::Rect<N> &bounds)
    {
      auto trans = transform.to_affine_transform<M, N>();
      return AccessorRW<T, N>(pr, fid, trans, bounds);
    }
  };

  template <typename OP, bool EXCLUSIVE, int32_t N>
  struct reduce_trans_accesor_fn {
    using Accessor = AccessorRD<OP, EXCLUSIVE, N>;
    template <int32_t M>
    Accessor operator()(const Legion::PhysicalRegion &pr,
                        Legion::FieldID fid,
                        const Transform &transform,
                        int32_t redop_id)
    {
      auto trans = transform.to_affine_transform<M, N>();
      return Accessor(pr, fid, redop_id, trans);
    }
    template <int32_t M>
    Accessor operator()(const Legion::PhysicalRegion &pr,
                        Legion::FieldID fid,
                        const Transform &transform,
                        int32_t redop_id,
                        const Legion::Rect<N> &bounds)
    {
      auto trans = transform.to_affine_transform<M, N>();
      return Accessor(pr, fid, redop_id, trans, bounds);
    }
  };

 public:
  template <typename T, int32_t DIM>
  AccessorRO<T, DIM> read_accessor() const;
  template <typename T, int32_t DIM>
  AccessorWO<T, DIM> write_accessor() const;
  template <typename T, int32_t DIM>
  AccessorRW<T, DIM> read_write_accessor() const;
  template <typename OP, bool EXCLUSIVE, int32_t DIM>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor() const;

 public:
  template <typename T, int32_t DIM>
  AccessorRO<T, DIM> read_accessor(const Legion::Rect<DIM> &bounds) const;
  template <typename T, int32_t DIM>
  AccessorWO<T, DIM> write_accessor(const Legion::Rect<DIM> &bounds) const;
  template <typename T, int32_t DIM>
  AccessorRW<T, DIM> read_write_accessor(const Legion::Rect<DIM> &bounds) const;
  template <typename OP, bool EXCLUSIVE, int32_t DIM>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor(const Legion::Rect<DIM> &bounds) const;

 public:
  template <int32_t DIM>
  Legion::Rect<DIM> shape() const;

 private:
  int32_t dim_{-1};
  int32_t redop_id_{-1};
  Legion::PhysicalRegion pr_{};
  Legion::FieldID fid_{-1U};
  Transform transform_{};

 private:
  bool readable_{false};
  bool writable_{false};
  bool reducible_{false};
};

class Array {
 public:
  Array() {}
  Array(int32_t dim, LegateTypeCode code, Legion::Future future);
  Array(int32_t dim, LegateTypeCode code, RegionField &&region_field);

 public:
  Array(Array &&other) noexcept;
  Array &operator=(Array &&other) noexcept;

 private:
  Array(const Array &other) = delete;
  Array &operator=(const Array &other) = delete;

 public:
  int32_t dim() const { return dim_; }
  LegateTypeCode code() const { return code_; }

 public:
  template <typename T, int32_t DIM>
  AccessorRO<T, DIM> read_accessor() const;
  template <typename T, int32_t DIM>
  AccessorWO<T, DIM> write_accessor() const;
  template <typename T, int32_t DIM>
  AccessorRW<T, DIM> read_write_accessor() const;
  template <typename OP, bool EXCLUSIVE, int32_t DIM>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor() const;

 public:
  template <typename T, int32_t DIM>
  AccessorRO<T, DIM> read_accessor(const Legion::Rect<DIM> &bounds) const;
  template <typename T, int32_t DIM>
  AccessorWO<T, DIM> write_accessor(const Legion::Rect<DIM> &bounds) const;
  template <typename T, int32_t DIM>
  AccessorRW<T, DIM> read_write_accessor(const Legion::Rect<DIM> &bounds) const;
  template <typename OP, bool EXCLUSIVE, int32_t DIM>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor(const Legion::Rect<DIM> &bounds) const;

 public:
  template <int32_t DIM>
  Legion::Rect<DIM> shape() const;

 public:
  UntypedScalar scalar() const;

 private:
  bool is_future_{false};
  int32_t dim_{-1};
  LegateTypeCode code_{MAX_TYPE_NUMBER};
  Legion::Future future_;
  RegionField region_field_;
};

}  // namespace numpy
}  // namespace legate

#include "core.inl"
