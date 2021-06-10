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
#include <type_traits>
#if defined(LEGATE_USE_CUDA) && defined(__CUDACC__)
#include "cuda_help.h"
#endif
#ifdef LEGATE_USE_OPENMP
#include <omp.h>
#endif

namespace legate {
namespace numpy {

// This is a small helper class that will also work if we have zero-sized arrays
// We also need to have this instead of std::array so that it works on devices
template <int DIM>
class Pitches {
 public:
  __CUDA_HD__
  inline size_t flatten(const Legion::Rect<DIM + 1>& rect)
  {
    size_t pitch  = 1;
    size_t volume = 1;
    for (int d = DIM; d >= 0; --d) {
      // Quick exit for empty rectangle dimensions
      if (rect.lo[d] > rect.hi[d]) return 0;
      const size_t diff = rect.hi[d] - rect.lo[d] + 1;
      volume *= diff;
      if (d > 0) {
        pitch *= diff;
        pitches[d - 1] = pitch;
      }
    }
    return volume;
  }
  __CUDA_HD__
  inline Legion::Point<DIM + 1> unflatten(size_t index, const Legion::Point<DIM + 1>& lo) const
  {
    Legion::Point<DIM + 1> point = lo;
    for (int d = 0; d < DIM; d++) {
      point[d] += index / pitches[d];
      index = index % pitches[d];
    }
    point[DIM] += index;
    return point;
  }

 private:
  size_t pitches[DIM];
};
// Specialization for the zero-sized case
template <>
class Pitches<0> {
 public:
  __CUDA_HD__
  inline size_t flatten(const Legion::Rect<1>& rect)
  {
    if (rect.lo[0] > rect.hi[0])
      return 0;
    else
      return (rect.hi[0] - rect.lo[0] + 1);
  }
  __CUDA_HD__
  inline Legion::Point<1> unflatten(size_t index, const Legion::Point<1>& lo) const
  {
    Legion::Point<1> point = lo;
    point[0] += index;
    return point;
  }
};

template <int DIM>
class CPULoop {
 public:
  static_assert(DIM <= 0, "Need more Looper instantiations");
};

// A small helper class that provides array index syntax for scalar
// value when abstracting over loop classes, this should be optimized
// away completely by the compiler's optimization passes
template <typename T, int DIM>
class Scalar {
 public:
  Scalar(const T& v) : value(v) {}

 public:
  inline Scalar<T, DIM - 1> operator[](coord_t) const { return Scalar<T, DIM - 1>(value); }

 private:
  const T& value;
};

template <typename T>
class Scalar<T, 1> {
 public:
  Scalar(const T& v) : value(v) {}

 public:
  inline const T& operator[](coord_t) const { return value; }

 private:
  const T& value;
};

template <>
class CPULoop<1> {
 public:
  template <typename Function, typename T, typename T1, typename... Args>
  static inline void unary_loop(
    const Function& func, const T& out, const T1& in, const Legion::Rect<1>& rect, Args&&... args)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      out[x] = func(in[x], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename... Args>
  static inline void unary_inplace(const Function& func,
                                   const T& inout,
                                   const Legion::Rect<1>& rect,
                                   Args&&... args)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      inout[x] = func(inout[x], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename T1, typename T2, typename... Args>
  static inline void binary_loop(const Function& func,
                                 const T& out,
                                 const T1& in1,
                                 const T2& in2,
                                 const Legion::Rect<1>& rect,
                                 Args&&... args)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      out[x] = func(in1[x], in2[x], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename T1, typename... Args>
  static inline void binary_inplace(
    const Function& func, const T& inout, const T1& in, const Legion::Rect<1>& rect, Args&&... args)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      inout[x] = func(inout[x], in[x], std::forward<Args>(args)...);
  }
  template <typename ReductionOp, typename VAL, typename ACC>
  static inline void unary_reduction_loop(ReductionOp op,
                                          VAL& lhs,
                                          const Legion::Rect<1>& rect,
                                          const ACC& in)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x) ReductionOp::template fold<true>(lhs, in[x]);
  }
};

template <>
class CPULoop<2> {
 public:
  template <typename Function, typename T, typename T1, typename... Args>
  static inline void unary_loop(
    const Function& func, const T& out, const T1& in1, const Legion::Rect<2>& rect, Args&&... args)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        out[x][y] = func(in1[x][y], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename... Args>
  static inline void unary_inplace(const Function& func,
                                   const T& inout,
                                   const Legion::Rect<2>& rect,
                                   Args&&... args)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        inout[x][y] = func(inout[x][y], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename T1, typename T2, typename... Args>
  static inline void binary_loop(const Function& func,
                                 const T& out,
                                 const T1& in1,
                                 const T2& in2,
                                 const Legion::Rect<2>& rect,
                                 Args&&... args)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        out[x][y] = func(in1[x][y], in2[x][y], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename T1, typename... Args>
  static inline void binary_inplace(
    const Function& func, const T& inout, const T1& in, const Legion::Rect<2>& rect, Args&&... args)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        inout[x][y] = func(inout[x][y], in[x][y], std::forward<Args>(args)...);
  }
  template <typename ReductionOp, typename VAL, typename ACC>
  static inline void unary_reduction_loop(ReductionOp op,
                                          VAL& lhs,
                                          const Legion::Rect<2>& rect,
                                          const ACC& in)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        ReductionOp::template fold<true>(lhs, in[x][y]);
  }
};

template <>
class CPULoop<3> {
 public:
  template <typename Function, typename T, typename T1, typename... Args>
  static inline void unary_loop(
    const Function& func, const T& out, const T1& in1, const Legion::Rect<3>& rect, Args&&... args)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
          out[x][y][z] = func(in1[x][y][z], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename... Args>
  static inline void unary_inplace(const Function& func,
                                   const T& inout,
                                   const Legion::Rect<3>& rect,
                                   Args&&... args)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
          inout[x][y][z] = func(inout[x][y][z], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename T1, typename T2, typename... Args>
  static inline void binary_loop(const Function& func,
                                 const T& out,
                                 const T1& in1,
                                 const T2& in2,
                                 const Legion::Rect<3>& rect,
                                 Args&&... args)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
          out[x][y][z] = func(in1[x][y][z], in2[x][y][z], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename T1, typename... Args>
  static inline void binary_inplace(
    const Function& func, const T& inout, const T1& in, const Legion::Rect<3>& rect, Args&&... args)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
          inout[x][y][z] = func(inout[x][y][z], in[x][y][z], std::forward<Args>(args)...);
  }
  template <typename ReductionOp, typename VAL, typename ACC>
  static inline void unary_reduction_loop(ReductionOp op,
                                          VAL& lhs,
                                          const Legion::Rect<3>& rect,
                                          const ACC& in)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
          ReductionOp::template fold<true>(lhs, in[x][y][z]);
  }
};

template <>
class CPULoop<4> {
 public:
  template <typename Function, typename T, typename T1, typename... Args>
  static inline void unary_loop(
    const Function& func, const T& out, const T1& in1, const Legion::Rect<4>& rect, Args&&... args)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
          for (int w = rect.lo[3]; w <= rect.hi[3]; ++w)
            out[x][y][z][w] = func(in1[x][y][z][w], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename... Args>
  static inline void unary_inplace(const Function& func,
                                   const T& inout,
                                   const Legion::Rect<4>& rect,
                                   Args&&... args)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
          for (int w = rect.lo[3]; w <= rect.hi[3]; ++w)
            inout[x][y][z][w] = func(inout[x][y][z][w], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename T1, typename T2, typename... Args>
  static inline void binary_loop(const Function& func,
                                 const T& out,
                                 const T1& in1,
                                 const T2& in2,
                                 const Legion::Rect<4>& rect,
                                 Args&&... args)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
          for (int w = rect.lo[3]; w <= rect.hi[3]; ++w)
            out[x][y][z][w] = func(in1[x][y][z][w], in2[x][y][z][w], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename T1, typename... Args>
  static inline void binary_inplace(
    const Function& func, const T& inout, const T1& in, const Legion::Rect<4>& rect, Args&&... args)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
          for (int w = rect.lo[3]; w <= rect.hi[3]; ++w)
            inout[x][y][z][w] =
              func(inout[x][y][z][w], in[x][y][z][w], std::forward<Args>(args)...);
  }
  template <typename ReductionOp, typename VAL, typename ACC>
  static inline void unary_reduction_loop(ReductionOp op,
                                          VAL& lhs,
                                          const Legion::Rect<4>& rect,
                                          const ACC& in)
  {
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
          for (int w = rect.lo[3]; w <= rect.hi[3]; ++w)
            ReductionOp::template fold<true>(lhs, in[x][y][z][w]);
  }
};

#ifdef LEGATE_USE_OPENMP
template <int DIM>
class OMPLoop {
 public:
  static_assert(DIM <= 0, "Need more OmpLooper instantiations");
};

template <>
class OMPLoop<1> {
 public:
  template <typename Function, typename T, typename T1, typename... Args>
  static inline void unary_loop(
    const Function& func, const T& out, const T1& in, const Legion::Rect<1>& rect, Args&&... args)
  {
#pragma omp parallel for schedule(static)
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      out[x] = func(in[x], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename... Args>
  static inline void unary_inplace(const Function& func,
                                   const T& inout,
                                   const Legion::Rect<1>& rect,
                                   Args&&... args)
  {
#pragma omp parallel for schedule(static)
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      inout[x] = func(inout[x], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename T1, typename T2, typename... Args>
  static inline void binary_loop(const Function& func,
                                 const T& out,
                                 const T1& in1,
                                 const T2& in2,
                                 const Legion::Rect<1>& rect,
                                 Args&&... args)
  {
#pragma omp parallel for schedule(static)
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      out[x] = func(in1[x], in2[x], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename T1, typename... Args>
  static inline void binary_inplace(
    const Function& func, const T& inout, const T1& in, const Legion::Rect<1>& rect, Args&&... args)
  {
#pragma omp parallel for schedule(static)
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      inout[x] = func(inout[x], in[x], std::forward<Args>(args)...);
  }
  template <typename ReductionOp, typename VAL, typename ACC>
  static inline void unary_reduction_loop(ReductionOp op,
                                          VAL* lhs,
                                          const Legion::Rect<1>& rect,
                                          const ACC& in)
  {
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
      for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
        ReductionOp::template fold<true>(lhs[tid], in[x]);
    }
  }
};

template <>
class OMPLoop<2> {
 public:
  template <typename Function, typename T, typename T1, typename... Args>
  static inline void unary_loop(
    const Function& func, const T& out, const T1& in1, const Legion::Rect<2>& rect, Args&&... args)
  {
#pragma omp parallel for schedule(static), collapse(2)
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        out[x][y] = func(in1[x][y], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename... Args>
  static inline void unary_inplace(const Function& func,
                                   const T& inout,
                                   const Legion::Rect<2>& rect,
                                   Args&&... args)
  {
#pragma omp parallel for schedule(static), collapse(2)
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        inout[x][y] = func(inout[x][y], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename T1, typename T2, typename... Args>
  static inline void binary_loop(const Function& func,
                                 const T& out,
                                 const T1& in1,
                                 const T2& in2,
                                 const Legion::Rect<2>& rect,
                                 Args&&... args)
  {
#pragma omp parallel for schedule(static), collapse(2)
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        out[x][y] = func(in1[x][y], in2[x][y], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename T1, typename... Args>
  static inline void binary_inplace(
    const Function& func, const T& inout, const T1& in, const Legion::Rect<2>& rect, Args&&... args)
  {
#pragma omp parallel for schedule(static), collapse(2)
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        inout[x][y] = func(inout[x][y], in[x][y], std::forward<Args>(args)...);
  }
  template <typename ReductionOp, typename VAL, typename ACC>
  static inline void unary_reduction_loop(ReductionOp op,
                                          VAL* lhs,
                                          const Legion::Rect<2>& rect,
                                          const ACC& in)
  {
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
#pragma omp for schedule(static), collapse(2)
      for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
        for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
          ReductionOp::template fold<true>(lhs[tid], in[x][y]);
    }
  }
};

template <>
class OMPLoop<3> {
 public:
  template <typename Function, typename T, typename T1, typename... Args>
  static inline void unary_loop(
    const Function& func, const T& out, const T1& in1, const Legion::Rect<3>& rect, Args&&... args)
  {
#pragma omp parallel for schedule(static), collapse(3)
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
          out[x][y][z] = func(in1[x][y][z], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename... Args>
  static inline void unary_inplace(const Function& func,
                                   const T& inout,
                                   const Legion::Rect<3>& rect,
                                   Args&&... args)
  {
#pragma omp parallel for schedule(static), collapse(3)
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
          inout[x][y][z] = func(inout[x][y][z], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename T1, typename T2, typename... Args>
  static inline void binary_loop(const Function& func,
                                 const T& out,
                                 const T1& in1,
                                 const T2& in2,
                                 const Legion::Rect<3>& rect,
                                 Args&&... args)
  {
#pragma omp parallel for schedule(static), collapse(3)
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
          out[x][y][z] = func(in1[x][y][z], in2[x][y][z], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename T1, typename... Args>
  static inline void binary_inplace(
    const Function& func, const T& inout, const T1& in, const Legion::Rect<3>& rect, Args&&... args)
  {
#pragma omp parallel for schedule(static), collapse(3)
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
          inout[x][y][z] = func(inout[x][y][z], in[x][y][z], std::forward<Args>(args)...);
  }
  template <typename ReductionOp, typename VAL, typename ACC>
  static inline void unary_reduction_loop(ReductionOp op,
                                          VAL* lhs,
                                          const Legion::Rect<3>& rect,
                                          const ACC& in)
  {
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
#pragma omp for schedule(static), collapse(3)
      for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
        for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
          for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
            ReductionOp::template fold<true>(lhs[tid], in[x][y][z]);
    }
  }
};

template <>
class OMPLoop<4> {
 public:
  template <typename Function, typename T, typename T1, typename... Args>
  static inline void unary_loop(
    const Function& func, const T& out, const T1& in1, const Legion::Rect<4>& rect, Args&&... args)
  {
#pragma omp parallel for schedule(static), collapse(4)
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
          for (int w = rect.lo[3]; w <= rect.hi[3]; ++w)
            out[x][y][z][w] = func(in1[x][y][z][w], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename... Args>
  static inline void unary_inplace(const Function& func,
                                   const T& inout,
                                   const Legion::Rect<4>& rect,
                                   Args&&... args)
  {
#pragma omp parallel for schedule(static), collapse(4)
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
          for (int w = rect.lo[3]; w <= rect.hi[3]; ++w)
            inout[x][y][z][w] = func(inout[x][y][z][w], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename T1, typename T2, typename... Args>
  static inline void binary_loop(const Function& func,
                                 const T& out,
                                 const T1& in1,
                                 const T2& in2,
                                 const Legion::Rect<4>& rect,
                                 Args&&... args)
  {
#pragma omp parallel for schedule(static), collapse(4)
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
          for (int w = rect.lo[3]; w <= rect.hi[3]; ++w)
            out[x][y][z][w] = func(in1[x][y][z][w], in2[x][y][z][w], std::forward<Args>(args)...);
  }
  template <typename Function, typename T, typename T1, typename... Args>
  static inline void binary_inplace(
    const Function& func, const T& inout, const T1& in, const Legion::Rect<4>& rect, Args&&... args)
  {
#pragma omp parallel for schedule(static), collapse(4)
    for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
      for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
        for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
          for (int w = rect.lo[3]; w <= rect.hi[3]; ++w)
            inout[x][y][z][w] =
              func(inout[x][y][z][w], in[x][y][z][w], std::forward<Args>(args)...);
  }
  template <typename ReductionOp, typename VAL, typename ACC>
  static inline void unary_reduction_loop(ReductionOp op,
                                          VAL* lhs,
                                          const Legion::Rect<4>& rect,
                                          const ACC& in)
  {
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
#pragma omp for schedule(static), collapse(4)
      for (int x = rect.lo[0]; x <= rect.hi[0]; ++x)
        for (int y = rect.lo[1]; y <= rect.hi[1]; ++y)
          for (int z = rect.lo[2]; z <= rect.hi[2]; ++z)
            for (int w = rect.lo[3]; w <= rect.hi[3]; ++w)
              ReductionOp::template fold<true>(lhs[tid], in[x][y][z][w]);
    }
  }
};
#endif

}  // namespace numpy
}  // namespace legate
