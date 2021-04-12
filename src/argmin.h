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

#ifndef __NUMPY_ARGMIN_H__
#define __NUMPY_ARGMIN_H__

#include "arg.h"
#include "numpy.h"

namespace legate {
namespace numpy {

template<typename T>
class ArgminReduction {
  // Empty definition
  // Specializations provided for each type
};

template<>
class ArgminReduction<__half> {
public:
  typedef Argval<__half> LHS;
  typedef Argval<__half> RHS;

  static const Argval<__half> identity;
  static const int            REDOP_ID = NUMPY_ARGMIN_REDOP * MAX_TYPE_NUMBER + HALF_LT;

  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs) {
    lhs.template apply<Legion::MinReduction<__half>, EXCLUSIVE>(rhs);
  }
  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2) {
    rhs1.template apply<Legion::MinReduction<__half>, EXCLUSIVE>(rhs2);
  }
};

template<>
class ArgminReduction<float> {
public:
  typedef Argval<float> LHS;
  typedef Argval<float> RHS;

  static const Argval<float> identity;
  static const int           REDOP_ID = NUMPY_ARGMIN_REDOP * MAX_TYPE_NUMBER + FLOAT_LT;

  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs) {
    lhs.template apply<Legion::MinReduction<float>, EXCLUSIVE>(rhs);
  }
  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2) {
    rhs1.template apply<Legion::MinReduction<float>, EXCLUSIVE>(rhs2);
  }
};

template<>
class ArgminReduction<double> {
public:
  typedef Argval<double> LHS;
  typedef Argval<double> RHS;

  static const Argval<double> identity;
  static const int            REDOP_ID = NUMPY_ARGMIN_REDOP * MAX_TYPE_NUMBER + DOUBLE_LT;

  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs) {
    lhs.template apply<Legion::MinReduction<double>, EXCLUSIVE>(rhs);
  }
  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2) {
    rhs1.template apply<Legion::MinReduction<double>, EXCLUSIVE>(rhs2);
  }
};

template<>
class ArgminReduction<int16_t> {
public:
  typedef Argval<int16_t> LHS;
  typedef Argval<int16_t> RHS;

  static const Argval<int16_t> identity;
  static const int             REDOP_ID = NUMPY_ARGMIN_REDOP * MAX_TYPE_NUMBER + INT16_LT;

  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs) {
    lhs.template apply<Legion::MinReduction<int16_t>, EXCLUSIVE>(rhs);
  }
  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2) {
    rhs1.template apply<Legion::MinReduction<int16_t>, EXCLUSIVE>(rhs2);
  }
};

template<>
class ArgminReduction<int32_t> {
public:
  typedef Argval<int32_t> LHS;
  typedef Argval<int32_t> RHS;

  static const Argval<int32_t> identity;
  static const int             REDOP_ID = NUMPY_ARGMIN_REDOP * MAX_TYPE_NUMBER + INT32_LT;

  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs) {
    lhs.template apply<Legion::MinReduction<int32_t>, EXCLUSIVE>(rhs);
  }
  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2) {
    rhs1.template apply<Legion::MinReduction<int32_t>, EXCLUSIVE>(rhs2);
  }
};

template<>
class ArgminReduction<int64_t> {
public:
  typedef Argval<int64_t> LHS;
  typedef Argval<int64_t> RHS;

  static const Argval<int64_t> identity;
  static const int             REDOP_ID = NUMPY_ARGMIN_REDOP * MAX_TYPE_NUMBER + INT64_LT;

  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs) {
    lhs.template apply<Legion::MinReduction<int64_t>, EXCLUSIVE>(rhs);
  }
  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2) {
    rhs1.template apply<Legion::MinReduction<int64_t>, EXCLUSIVE>(rhs2);
  }
};

template<>
class ArgminReduction<uint16_t> {
public:
  typedef Argval<uint16_t> LHS;
  typedef Argval<uint16_t> RHS;

  static const Argval<uint16_t> identity;
  static const int              REDOP_ID = NUMPY_ARGMIN_REDOP * MAX_TYPE_NUMBER + UINT16_LT;

  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs) {
    lhs.template apply<Legion::MinReduction<uint16_t>, EXCLUSIVE>(rhs);
  }
  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2) {
    rhs1.template apply<Legion::MinReduction<uint16_t>, EXCLUSIVE>(rhs2);
  }
};

template<>
class ArgminReduction<uint32_t> {
public:
  typedef Argval<uint32_t> LHS;
  typedef Argval<uint32_t> RHS;

  static const Argval<uint32_t> identity;
  static const int              REDOP_ID = NUMPY_ARGMIN_REDOP * MAX_TYPE_NUMBER + UINT32_LT;

  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs) {
    lhs.template apply<Legion::MinReduction<uint32_t>, EXCLUSIVE>(rhs);
  }
  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2) {
    rhs1.template apply<Legion::MinReduction<uint32_t>, EXCLUSIVE>(rhs2);
  }
};

template<>
class ArgminReduction<uint64_t> {
public:
  typedef Argval<uint64_t> LHS;
  typedef Argval<uint64_t> RHS;

  static const Argval<uint64_t> identity;
  static const int              REDOP_ID = NUMPY_ARGMIN_REDOP * MAX_TYPE_NUMBER + UINT64_LT;

  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs) {
    lhs.template apply<Legion::MinReduction<uint64_t>, EXCLUSIVE>(rhs);
  }
  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2) {
    rhs1.template apply<Legion::MinReduction<uint64_t>, EXCLUSIVE>(rhs2);
  }
};

template<>
class ArgminReduction<bool> {
public:
  typedef Argval<bool> LHS;
  typedef Argval<bool> RHS;

  static const Argval<bool> identity;
  static const int          REDOP_ID = NUMPY_ARGMIN_REDOP * MAX_TYPE_NUMBER + BOOL_LT;

  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs) {
    lhs.template apply<Legion::MinReduction<bool>, EXCLUSIVE>(rhs);
  }
  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2) {
    rhs1.template apply<Legion::MinReduction<bool>, EXCLUSIVE>(rhs2);
  }
};

// template<>
// class ArgminReduction<complex<__half>> {
// public:
//   typedef Argval<complex<__half>> LHS;
//   typedef Argval<complex<__half>> RHS;

//   static const Argval<complex<__half>> identity;
//   static const int                     REDOP_ID = NUMPY_ARGMIN_REDOP * MAX_TYPE_NUMBER + COMPLEX64_LT;

//   template<bool EXCLUSIVE>
//   __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs) {
//     lhs.template apply<Legion::MinReduction<complex<__half>>, EXCLUSIVE>(rhs);
//   }
//   template<bool EXCLUSIVE>
//   __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2) {
//     rhs1.template apply<Legion::MinReduction<complex<__half>>, EXCLUSIVE>(rhs2);
//   }
// };

template<>
class ArgminReduction<complex<float>> {
public:
  typedef Argval<complex<float>> LHS;
  typedef Argval<complex<float>> RHS;

  static const Argval<complex<float>> identity;
  static const int                    REDOP_ID = NUMPY_ARGMIN_REDOP * MAX_TYPE_NUMBER + COMPLEX64_LT;

  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs) {
    lhs.template apply<Legion::MinReduction<complex<float>>, EXCLUSIVE>(rhs);
  }
  template<bool EXCLUSIVE>
  __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2) {
    rhs1.template apply<Legion::MinReduction<complex<float>>, EXCLUSIVE>(rhs2);
  }
};

// TBD: when complex<double> gets a reduction
// template<>
// class ArgminReduction<complex<double>> {
// public:
//   typedef Argval<complex<double>> LHS;
//   typedef Argval<complex<double>> RHS;

//   static const Argval<complex<double>> identity;
//   static const int                     REDOP_ID = NUMPY_ARGMIN_REDOP * MAX_TYPE_NUMBER + COMPLEX128_LT;

//   template<bool EXCLUSIVE>
//   __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs) {
//     lhs.template apply<Legion::MinReduction<complex<double>>, EXCLUSIVE>(rhs);
//   }
//   template<bool EXCLUSIVE>
//   __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2) {
//     rhs1.template apply<Legion::MinReduction<complex<double>>, EXCLUSIVE>(rhs2);
//   }
// };

// For doing argmin into particular dimension
template<typename T>
class ArgminTask : public NumPyTask<ArgminTask<T>> {
public:
  static const int TASK_ID;
  static const int REGIONS = 2;
#if 0
    public:
      template<typename TASK>
      static void set_layout_constraints(LegateVariant variant, 
                  TaskLayoutConstraintSet &layout_constraints);
#endif
public:
  static void cpu_variant(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
                          Legion::Runtime* runtime);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
                          Legion::Runtime* runtime);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
                          Legion::Runtime* runtime);
#endif
};

// For doing a reduction to a single value
template<typename T>
class ArgminReducTask : public NumPyTask<ArgminReducTask<T>> {
public:
  static const int TASK_ID;
  static const int REGIONS = 1;

public:
  static Argval<T> cpu_variant(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
                               Legion::Runtime* runtime);
#ifdef LEGATE_USE_OPENMP
  static Argval<T> omp_variant(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
                               Legion::Runtime* runtime);
#endif
#ifdef LEGATE_USE_CUDA
  static Legion::DeferredReduction<ArgminReduction<T>> gpu_variant(const Legion::Task*                        task,
                                                                   const std::vector<Legion::PhysicalRegion>& regions,
                                                                   Legion::Context ctx, Legion::Runtime* runtime);
#endif
};

template<typename T>
class ArgminRadixTask : public NumPyTask<ArgminRadixTask<T>> {
public:
  static const int TASK_ID;
  static const int REGIONS = MAX_REDUCTION_RADIX;

public:
  static void cpu_variant(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
                          Legion::Runtime* runtime);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
                          Legion::Runtime* runtime);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
                          Legion::Runtime* runtime);
#endif
};

}    // namespace numpy
}    // namespace legate

#endif    // __NUMPY_ARGMIN_H__
