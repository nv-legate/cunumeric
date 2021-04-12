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

#include "dot.h"
#include "proj.h"
#include <cblas.h>
#ifdef LEGATE_USE_OPENMP
#  include <alloca.h>
#  include <omp.h>
#endif

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T>
template<typename TASK>
/*static*/ void DotTask<T>::set_layout_constraints(LegateVariant variant, TaskLayoutConstraintSet& layout_constraints) {
  // Don't put constraints on the first region requirement as it
  // could either be a reduction instance or a normal instance
  // depending on whether we are doing an index space launch or not
  for (int idx = 1; idx < TASK::REGIONS; idx++)
    layout_constraints.add_layout_constraint(idx, Core::get_soa_layout());
}

static inline void __convert_half_vector_to_float(const __half* ptr, float* out, size_t n) {
  for (unsigned idx = 0; idx < n; idx++)
    out[idx] = ptr[idx];
}

static inline void __convert_half_matrix_to_float(const __half* ptr, float* out, size_t m, size_t n, size_t pitch) {
  for (unsigned i = 0; i < m; i++)
    for (unsigned j = 0; j < n; j++)
      out[i * n + j] = ptr[i * pitch + j];
}

static inline void __convert_float_vector_to_half(const float* ptr, __half* out, size_t n) {
  for (unsigned idx = 0; idx < n; idx++)
    out[idx] = ptr[idx];
}

static inline void __convert_float_matrix_to_half(const float* ptr, __half* out, size_t m, size_t n, size_t pitch) {
  for (unsigned i = 0; i < m; i++)
    for (unsigned j = 0; j < n; j++)
      out[i * pitch + j] = ptr[i * n + j];
}

static void dot_half(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const bool         partial   = derez.unpack_bool();
  const int          extra_dim = derez.unpack_dimension();
  const int          dim       = derez.unpack_dimension();
  float*             temp_in1  = NULL;
  float*             temp_in2  = NULL;
  float*             temp_out  = NULL;
  switch (dim) {
    case 1: {
      // This has to be matrix vector
      const Rect<1> out_rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (out_rect.empty()) return;
      void* out_ptr;
#if 1
      const int reduce = (task->regions[0].privilege == READ_WRITE) ? 1 : 0;
      if (partial) {
        if (reduce == 1) {
          const AccessorRW<float, 1> out =
              (extra_dim >= 0)
                  ? derez.unpack_accessor_RW<float, 1>(regions[0], out_rect, 1 /*out extra dim*/, task->index_point[extra_dim])
                  : derez.unpack_accessor_RW<float, 1>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect);
        } else {
          const AccessorWO<float, 1> out =
              (extra_dim >= 0)
                  ? derez.unpack_accessor_WO<float, 1>(regions[0], out_rect, 1 /*out extra dim*/, task->index_point[extra_dim])
                  : derez.unpack_accessor_WO<float, 1>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect);
        }
      } else {
        if (reduce == 1) {
          const AccessorRW<__half, 1> out =
              (extra_dim >= 0)
                  ? derez.unpack_accessor_RW<__half, 1>(regions[0], out_rect, 1 /*out extra dim*/, task->index_point[extra_dim])
                  : derez.unpack_accessor_RW<__half, 1>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect);
        } else {
          const AccessorWO<__half, 1> out =
              (extra_dim >= 0)
                  ? derez.unpack_accessor_WO<__half, 1>(regions[0], out_rect, 1 /*out extra dim*/, task->index_point[extra_dim])
                  : derez.unpack_accessor_WO<__half, 1>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect);
        }
      }
#else
      const int reduce = task->is_index_space ? 1 : 0;
      if (partial) {
        if (task->is_index_space) {
          const AccessorRD<SumReduction<float>, true /*exclusive*/, 1> out =
              derez.unpack_accessor_RD<SumReduction<float>, true, 1>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect);
        } else {
          const AccessorWO<float, 1> out = derez.unpack_accessor_WO<float, 1>(regions[0], out_rect);
          out_ptr                        = out.ptr(out_rect);
        }
      } else {
        if (task->is_index_space) {
          const AccessorRD<SumReduction<__half>, true /*exclusive*/, 1> out =
              derez.unpack_accessor_RD<SumReduction<__half>, true, 1>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect);
        } else {
          const AccessorWO<__half, 1> out = derez.unpack_accessor_WO<__half, 1>(regions[0], out_rect);
          out_ptr                         = out.ptr(out_rect);
        }
      }
#endif
      const int dim1 = derez.unpack_dimension();
      if (dim1 == 1) {
        const Rect<1> in1_rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
        if (in1_rect.empty()) return;
        const AccessorRO<__half, 1> in1     = derez.unpack_accessor_RO<__half, 1>(regions[1], in1_rect);
        const __half*               in1_ptr = in1.ptr(in1_rect);

        const int dim2 = derez.unpack_dimension();
        assert(dim2 == 2);
        const Rect<2> in2_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
        if (in2_rect.empty()) return;
        const AccessorRO<__half, 2> in2 = derez.unpack_accessor_RO<__half, 2>(regions[2], in2_rect);
        // Construct the rect we actually want to do the math for
        const Rect<2> act_rect(Point<2>(in1_rect.lo[0], out_rect.lo[0]), Point<2>(in1_rect.hi[0], out_rect.hi[0]));
        assert(in2_rect.contains(act_rect));
        size_t        in2_strides[2];
        const __half* in2_ptr = in2.ptr(act_rect, in2_strides);
        const coord_t m       = (act_rect.hi[0] - act_rect.lo[0]) + 1;
        const coord_t n       = (act_rect.hi[1] - act_rect.lo[1]) + 1;

        temp_in1 = (float*)malloc(m * sizeof(float));
        __convert_half_vector_to_float(in1_ptr, temp_in1, m);
        temp_in2 = (float*)malloc(n * m * sizeof(float));
        __convert_half_matrix_to_float(in2_ptr, temp_in2, n, m, in2_strides[0]);

        if (partial) {
          // We can do it in-place because the output is 32-bit floats
          cblas_sgemv(CblasRowMajor, CblasTrans, m, n, 1, temp_in2, m, temp_in1, 1, reduce, (float*)out_ptr, 1);
        } else {
          // Make a tempory output
          temp_out = (float*)malloc(n * sizeof(float));
          cblas_sgemv(CblasRowMajor, CblasTrans, m, n, 1, temp_in2, m, temp_in1, 1, reduce, temp_out, 1);
          __convert_float_vector_to_half(temp_out, (__half*)out_ptr, n);
        }
      } else {
        assert(dim1 == 2);
        const Rect<2> in1_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
        if (in1_rect.empty()) return;
        const AccessorRO<__half, 2> in1 = derez.unpack_accessor_RO<__half, 2>(regions[1], in1_rect);

        const int dim2 = derez.unpack_dimension();
        assert(dim2 == 1);
        const Rect<1> in2_rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
        if (in2_rect.empty()) return;
        const AccessorRO<__half, 1> in2     = derez.unpack_accessor_RO<__half, 1>(regions[2], in2_rect);
        const __half*               in2_ptr = in2.ptr(in2_rect);

        // Construct the rect we actually want to do the math for
        const Rect<2> act_rect(Point<2>(out_rect.lo[0], in2_rect.lo[0]), Point<2>(out_rect.hi[0], in2_rect.hi[0]));
        assert(in1_rect.contains(act_rect));
        size_t        in1_strides[2];
        const __half* in1_ptr = in1.ptr(act_rect, in1_strides);
        const coord_t m       = (act_rect.hi[0] - act_rect.lo[0]) + 1;
        const coord_t n       = (act_rect.hi[1] - act_rect.lo[1]) + 1;

        temp_in1 = (float*)malloc(m * n * sizeof(float));
        __convert_half_matrix_to_float(in1_ptr, temp_in1, m, n, in1_strides[0]);
        temp_in2 = (float*)malloc(n * sizeof(float));
        __convert_half_vector_to_float(in2_ptr, temp_in2, n);

        if (partial) {
          // We can do it in-place because the output is 32-bit floats
          cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1, temp_in1, n, temp_in2, 1, reduce, (float*)out_ptr, 1);
        } else {
          // Make a temporary output
          temp_out = (float*)malloc(m * sizeof(float));
          cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1, temp_in1, n, temp_in2, 1, reduce, temp_out, 1);
          __convert_float_vector_to_half(temp_out, (__half*)out_ptr, m);
        }
      }
      break;
    }
    case 2: {
      // This has to be matrix multiply for us right now
      const int     reduce   = (task->regions[0].privilege == READ_WRITE) ? 1 : 0;
      const Rect<2> out_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (out_rect.empty()) return;
      void*  out_ptr;
      size_t out_strides[2];
      if (partial) {
        if (task->regions[0].privilege == READ_WRITE) {
          const AccessorRW<float, 2> out =
              (extra_dim >= 0) ? derez.unpack_accessor_RW<float, 2>(regions[0], out_rect, extra_dim, task->index_point[extra_dim])
                               : derez.unpack_accessor_RW<float, 2>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect, out_strides);
        } else {
          const AccessorWO<float, 2> out =
              (extra_dim >= 0) ? derez.unpack_accessor_WO<float, 2>(regions[0], out_rect, extra_dim, task->index_point[extra_dim])
                               : derez.unpack_accessor_WO<float, 2>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect, out_strides);
        }
      } else {
        if (task->regions[0].privilege == READ_WRITE) {
          const AccessorRW<__half, 2> out =
              (extra_dim >= 0) ? derez.unpack_accessor_RW<__half, 2>(regions[0], out_rect, extra_dim, task->index_point[extra_dim])
                               : derez.unpack_accessor_RW<__half, 2>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect, out_strides);
        } else {
          const AccessorWO<__half, 2> out =
              (extra_dim >= 0) ? derez.unpack_accessor_WO<__half, 2>(regions[0], out_rect, extra_dim, task->index_point[extra_dim])
                               : derez.unpack_accessor_WO<__half, 2>(regions[0], out_rect);
          out_ptr = out.ptr(out_rect, out_strides);
        }
      }
      const coord_t m = (out_rect.hi[0] - out_rect.lo[0]) + 1;
      const coord_t n = (out_rect.hi[1] - out_rect.lo[1]) + 1;

      const int dim1 = derez.unpack_dimension();
      assert(dim1 == 2);
      const Rect<2> in1_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (in1_rect.empty()) return;
      const AccessorRO<__half, 2> in1 = derez.unpack_accessor_RO<__half, 2>(regions[1], in1_rect);
      size_t                      in1_strides[2];
      const __half*               in1_ptr = in1.ptr(in1_rect, in1_strides);
      assert(m == ((in1_rect.hi[0] - in1_rect.lo[0]) + 1));
      const coord_t k = (in1_rect.hi[1] - in1_rect.lo[1]) + 1;

      const int dim2 = derez.unpack_dimension();
      assert(dim2 == 2);
      const Rect<2> in2_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (in2_rect.empty()) return;
      const AccessorRO<__half, 2> in2 = derez.unpack_accessor_RO<__half, 2>(regions[2], in2_rect);
      size_t                      in2_strides[2];
      const __half*               in2_ptr = in2.ptr(in2_rect, in2_strides);
      assert(k == ((in2_rect.hi[0] - in2_rect.lo[0]) + 1));
      assert(n == ((in2_rect.hi[1] - in2_rect.lo[1]) + 1));

      temp_in1 = (float*)malloc(m * k * sizeof(float));
      __convert_half_matrix_to_float(in1_ptr, temp_in1, m, k, in1_strides[0]);
      temp_in2 = (float*)malloc(k * n * sizeof(float));
      __convert_half_matrix_to_float(in2_ptr, temp_in2, k, n, in2_strides[0]);

      if (partial) {
        // We can do this in-place since we know the output is 32-bit floats
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, temp_in1, k, temp_in2, n, reduce, (float*)out_ptr,
                    out_strides[0]);
      } else {
        temp_out = (float*)malloc(m * n * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, temp_in1, k, temp_in2, n, reduce, temp_out, n);
        __convert_float_matrix_to_half(temp_out, (__half*)out_ptr, m, n, out_strides[0]);
      }
      break;
    }
    default:
      assert(false);    // we don't support any other updates
  }
  // Free any temporary arrays that we made
  if (temp_in1 != NULL) free(temp_in1);
  if (temp_in2 != NULL) free(temp_in2);
  if (temp_out != NULL) free(temp_out);
}

template<>
/*static*/ void DotTask<__half>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                             Runtime* runtime) {
  openblas_set_num_threads(1);    // make sure this isn't overzealous
  dot_half(task, regions, ctx, runtime);
}

#ifdef LEGATE_USE_OPENMP
template<>
/*static*/ void DotTask<__half>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                             Runtime* runtime) {
  openblas_set_num_threads(omp_get_max_threads());
  dot_half(task, regions, ctx, runtime);
}
#endif

static void dot_float(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          extra_dim = derez.unpack_dimension();
  const int          dim       = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      // This has to be matrix vector
      const Rect<1> out_rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (out_rect.empty()) return;
      float* out_ptr;
#if 1
      const int reduce = (task->regions[0].privilege == READ_WRITE) ? 1 : 0;
      if (reduce == 1) {
        const AccessorRW<float, 1> out =
            (extra_dim >= 0)
                ? derez.unpack_accessor_RW<float, 1>(regions[0], out_rect, 1 /*out extra dim*/, task->index_point[extra_dim])
                : derez.unpack_accessor_RW<float, 1>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect);
      } else {
        const AccessorWO<float, 1> out =
            (extra_dim >= 0)
                ? derez.unpack_accessor_WO<float, 1>(regions[0], out_rect, 1 /*out extra dim*/, task->index_point[extra_dim])
                : derez.unpack_accessor_WO<float, 1>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect);
      }
#else
      const int reduce = task->is_index_space ? 1 : 0;
      if (task->is_index_space) {
        const AccessorRD<SumReduction<float>, true /*exclusive*/, 1> out =
            derez.unpack_accessor_RD<SumReduction<float>, true, 1>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect);
      } else {
        const AccessorWO<float, 1> out = derez.unpack_accessor_WO<float, 1>(regions[0], out_rect);
        out_ptr                        = out.ptr(out_rect);
      }
#endif
      const int dim1 = derez.unpack_dimension();
      if (dim1 == 1) {
        const Rect<1> in1_rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
        if (in1_rect.empty()) return;
        const AccessorRO<float, 1> in1     = derez.unpack_accessor_RO<float, 1>(regions[1], in1_rect);
        const float*               in1_ptr = in1.ptr(in1_rect);

        const int dim2 = derez.unpack_dimension();
        assert(dim2 == 2);
        const Rect<2> in2_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
        if (in2_rect.empty()) return;
        const AccessorRO<float, 2> in2 = derez.unpack_accessor_RO<float, 2>(regions[2], in2_rect);
        // Construct the rect we actually want to do the math for
        const Rect<2> act_rect(Point<2>(in1_rect.lo[0], out_rect.lo[0]), Point<2>(in1_rect.hi[0], out_rect.hi[0]));
        assert(in2_rect.contains(act_rect));
        size_t        in2_strides[2];
        const float*  in2_ptr = in2.ptr(act_rect, in2_strides);
        const coord_t m       = (act_rect.hi[0] - act_rect.lo[0]) + 1;
        const coord_t n       = (act_rect.hi[1] - act_rect.lo[1]) + 1;

        cblas_sgemv(CblasRowMajor, CblasTrans, m, n, 1, in2_ptr, in2_strides[0], in1_ptr, 1, reduce, out_ptr, 1);
      } else {
        assert(dim1 == 2);
        const Rect<2> in1_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
        if (in1_rect.empty()) return;
        const AccessorRO<float, 2> in1 = derez.unpack_accessor_RO<float, 2>(regions[1], in1_rect);

        const int dim2 = derez.unpack_dimension();
        assert(dim2 == 1);
        const Rect<1> in2_rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
        if (in2_rect.empty()) return;
        const AccessorRO<float, 1> in2     = derez.unpack_accessor_RO<float, 1>(regions[2], in2_rect);
        const float*               in2_ptr = in2.ptr(in2_rect);

        // Construct the rect we actually want to do the math for
        const Rect<2> act_rect(Point<2>(out_rect.lo[0], in2_rect.lo[0]), Point<2>(out_rect.hi[0], in2_rect.hi[0]));
        assert(in1_rect.contains(act_rect));
        size_t        in1_strides[2];
        const float*  in1_ptr = in1.ptr(act_rect, in1_strides);
        const coord_t m       = (act_rect.hi[0] - act_rect.lo[0]) + 1;
        const coord_t n       = (act_rect.hi[1] - act_rect.lo[1]) + 1;

        cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1, in1_ptr, in1_strides[0], in2_ptr, 1, reduce, out_ptr, 1);
      }
      break;
    }
    case 2: {
      // This has to be matrix multiply for us right now
      const int     reduce   = (task->regions[0].privilege == READ_WRITE) ? 1 : 0;
      const Rect<2> out_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (out_rect.empty()) return;
      float* out_ptr;
      size_t out_strides[2];
      if (task->regions[0].privilege == READ_WRITE) {
        const AccessorRW<float, 2> out =
            (extra_dim >= 0) ? derez.unpack_accessor_RW<float, 2>(regions[0], out_rect, extra_dim, task->index_point[extra_dim])
                             : derez.unpack_accessor_RW<float, 2>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect, out_strides);
      } else {
        const AccessorWO<float, 2> out =
            (extra_dim >= 0) ? derez.unpack_accessor_WO<float, 2>(regions[0], out_rect, extra_dim, task->index_point[extra_dim])
                             : derez.unpack_accessor_WO<float, 2>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect, out_strides);
      }
      const coord_t m = (out_rect.hi[0] - out_rect.lo[0]) + 1;
      const coord_t n = (out_rect.hi[1] - out_rect.lo[1]) + 1;

      const int dim1 = derez.unpack_dimension();
      assert(dim1 == 2);
      const Rect<2> in1_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (in1_rect.empty()) return;
      const AccessorRO<float, 2> in1 = derez.unpack_accessor_RO<float, 2>(regions[1], in1_rect);
      size_t                     in1_strides[2];
      const float*               in1_ptr = in1.ptr(in1_rect, in1_strides);
      assert(m == ((in1_rect.hi[0] - in1_rect.lo[0]) + 1));
      const coord_t k = (in1_rect.hi[1] - in1_rect.lo[1]) + 1;

      const int dim2 = derez.unpack_dimension();
      assert(dim2 == 2);
      const Rect<2> in2_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (in2_rect.empty()) return;
      const AccessorRO<float, 2> in2 = derez.unpack_accessor_RO<float, 2>(regions[2], in2_rect);
      size_t                     in2_strides[2];
      const float*               in2_ptr = in2.ptr(in2_rect, in2_strides);
      assert(k == ((in2_rect.hi[0] - in2_rect.lo[0]) + 1));
      assert(n == ((in2_rect.hi[1] - in2_rect.lo[1]) + 1));

      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, in1_ptr, in1_strides[0], in2_ptr, in2_strides[0], reduce,
                  out_ptr, out_strides[0]);
      break;
    }
    default:
      assert(false);    // we don't support any other updates
  }
}

template<>
/*static*/ void DotTask<float>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                            Runtime* runtime) {
  openblas_set_num_threads(1);    // make sure this isn't overzealous
  dot_float(task, regions, ctx, runtime);
}

#ifdef LEGATE_USE_OPENMP
template<>
/*static*/ void DotTask<float>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                            Runtime* runtime) {
  openblas_set_num_threads(omp_get_max_threads());
  dot_float(task, regions, ctx, runtime);
}
#endif

static void dot_double(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          extra_dim = derez.unpack_dimension();
  const int          dim       = derez.unpack_dimension();
  switch (dim) {
    case 1: {
      // This has to be matrix vector
      const Rect<1> out_rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      if (out_rect.empty()) return;
      double* out_ptr;
#if 1
      const int reduce = (task->regions[0].privilege == READ_WRITE) ? 1 : 0;
      if (reduce == 1) {
        const AccessorRW<double, 1> out =
            (extra_dim >= 0)
                ? derez.unpack_accessor_RW<double, 1>(regions[0], out_rect, 1 /*out extra dim*/, task->index_point[extra_dim])
                : derez.unpack_accessor_RW<double, 1>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect);
      } else {
        const AccessorWO<double, 1> out =
            (extra_dim >= 0)
                ? derez.unpack_accessor_WO<double, 1>(regions[0], out_rect, 1 /*out extra dim*/, task->index_point[extra_dim])
                : derez.unpack_accessor_WO<double, 1>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect);
      }
#else
      const int reduce = task->is_index_space ? 1 : 0;
      if (task->is_index_space) {
        const AccessorRD<SumReduction<double>, true /*exclusive*/, 1> out =
            derez.unpack_accessor_RD<SumReduction<double>, true, 1>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect);
      } else {
        const AccessorWO<double, 1> out = derez.unpack_accessor_WO<double, 1>(regions[0], out_rect);
        out_ptr                         = out.ptr(out_rect);
      }
#endif
      const int dim1 = derez.unpack_dimension();
      if (dim1 == 1) {
        const Rect<1> in1_rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
        if (in1_rect.empty()) return;
        const AccessorRO<double, 1> in1     = derez.unpack_accessor_RO<double, 1>(regions[1], in1_rect);
        const double*               in1_ptr = in1.ptr(in1_rect);

        const int dim2 = derez.unpack_dimension();
        assert(dim2 == 2);
        const Rect<2> in2_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
        if (in2_rect.empty()) return;
        const AccessorRO<double, 2> in2 = derez.unpack_accessor_RO<double, 2>(regions[2], in2_rect);
        // Construct the rect we actually want to do the math for
        const Rect<2> act_rect(Point<2>(in1_rect.lo[0], out_rect.lo[0]), Point<2>(in1_rect.hi[0], out_rect.hi[0]));
        assert(in2_rect.contains(act_rect));
        size_t        in2_strides[2];
        const double* in2_ptr = in2.ptr(act_rect, in2_strides);
        const coord_t m       = (act_rect.hi[0] - act_rect.lo[0]) + 1;
        const coord_t n       = (act_rect.hi[1] - act_rect.lo[1]) + 1;

        cblas_dgemv(CblasRowMajor, CblasTrans, m, n, 1, in2_ptr, in2_strides[0], in1_ptr, 1, reduce, out_ptr, 1);
      } else {
        assert(dim1 == 2);
        const Rect<2> in1_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
        if (in1_rect.empty()) return;
        const AccessorRO<double, 2> in1 = derez.unpack_accessor_RO<double, 2>(regions[1], in1_rect);

        const int dim2 = derez.unpack_dimension();
        assert(dim2 == 1);
        const Rect<1> in2_rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
        if (in2_rect.empty()) return;
        const AccessorRO<double, 1> in2     = derez.unpack_accessor_RO<double, 1>(regions[2], in2_rect);
        const double*               in2_ptr = in2.ptr(in2_rect);

        // Construct the rect we actually want to do the math for
        const Rect<2> act_rect(Point<2>(out_rect.lo[0], in2_rect.lo[0]), Point<2>(out_rect.hi[0], in2_rect.hi[0]));
        assert(in1_rect.contains(act_rect));
        size_t        in1_strides[2];
        const double* in1_ptr = in1.ptr(act_rect, in1_strides);
        const coord_t m       = (act_rect.hi[0] - act_rect.lo[0]) + 1;
        const coord_t n       = (act_rect.hi[1] - act_rect.lo[1]) + 1;

        cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1, in1_ptr, in1_strides[0], in2_ptr, 1, reduce, out_ptr, 1);
      }
      break;
    }
    case 2: {
      // This has to be matrix multiply for us right now
      const int     reduce   = (task->regions[0].privilege == READ_WRITE) ? 1 : 0;
      const Rect<2> out_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (out_rect.empty()) return;
      double* out_ptr;
      size_t  out_strides[2];
      if (task->regions[0].privilege == READ_WRITE) {
        const AccessorRW<double, 2> out =
            (extra_dim >= 0) ? derez.unpack_accessor_RW<double, 2>(regions[0], out_rect, extra_dim, task->index_point[extra_dim])
                             : derez.unpack_accessor_RW<double, 2>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect, out_strides);
      } else {
        const AccessorWO<double, 2> out =
            (extra_dim >= 0) ? derez.unpack_accessor_WO<double, 2>(regions[0], out_rect, extra_dim, task->index_point[extra_dim])
                             : derez.unpack_accessor_WO<double, 2>(regions[0], out_rect);
        out_ptr = out.ptr(out_rect, out_strides);
      }
      const coord_t m = (out_rect.hi[0] - out_rect.lo[0]) + 1;
      const coord_t n = (out_rect.hi[1] - out_rect.lo[1]) + 1;

      const int dim1 = derez.unpack_dimension();
      assert(dim1 == 2);
      const Rect<2> in1_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (in1_rect.empty()) return;
      const AccessorRO<double, 2> in1 = derez.unpack_accessor_RO<double, 2>(regions[1], in1_rect);
      size_t                      in1_strides[2];
      const double*               in1_ptr = in1.ptr(in1_rect, in1_strides);
      assert(m == ((in1_rect.hi[0] - in1_rect.lo[0]) + 1));
      const coord_t k = (in1_rect.hi[1] - in1_rect.lo[1]) + 1;

      const int dim2 = derez.unpack_dimension();
      assert(dim2 == 2);
      const Rect<2> in2_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      if (in2_rect.empty()) return;
      const AccessorRO<double, 2> in2 = derez.unpack_accessor_RO<double, 2>(regions[2], in2_rect);
      size_t                      in2_strides[2];
      const double*               in2_ptr = in2.ptr(in2_rect, in2_strides);
      assert(k == ((in2_rect.hi[0] - in2_rect.lo[0]) + 1));
      assert(n == ((in2_rect.hi[1] - in2_rect.lo[1]) + 1));

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, in1_ptr, in1_strides[0], in2_ptr, in2_strides[0], reduce,
                  out_ptr, out_strides[0]);
      break;
    }
    default:
      assert(false);    // we don't support any other updates
  }
}

template<>
/*static*/ void DotTask<double>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                             Runtime* runtime) {
  openblas_set_num_threads(1);    // make sure this isn't overzealous
  dot_double(task, regions, ctx, runtime);
}

#ifdef LEGATE_USE_OPENMP
template<>
/*static*/ void DotTask<double>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                             Runtime* runtime) {
  openblas_set_num_threads(omp_get_max_threads());
  dot_double(task, regions, ctx, runtime);
}
#endif

template<typename T>
/*static*/ T DotReducTask<T>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                          Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  assert(dim == 1);
  const Rect<1>          rect   = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  const AccessorRO<T, 1> in1    = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
  const AccessorRO<T, 1> in2    = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
  T                      result = SumReduction<T>::identity;
  for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
    T temp = in1[x];
    ProdReduction<T>::template fold<true /*exclusive*/>(temp, in2[x]);
    SumReduction<T>::template fold<true /*exclusive*/>(result, temp);
  }
  return result;
}

#ifdef LEGATE_USE_OPENMP
template<typename T>
/*static*/ T DotReducTask<T>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                          Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  assert(dim == 1);
  const Rect<1>          rect        = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  const AccessorRO<T, 1> in1         = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
  const AccessorRO<T, 1> in2         = derez.unpack_accessor_RO<T, 1>(regions[1], rect);
  const int              max_threads = omp_get_max_threads();
  T*                     results     = (T*)alloca(max_threads * sizeof(T));
  for (int i = 0; i < max_threads; i++)
    results[i] = SumReduction<T>::identity;
#  pragma omp parallel
  {
    const int tid = omp_get_thread_num();
#  pragma omp for nowait
    for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++) {
      T temp = in1[x];
      ProdReduction<T>::template fold<true /*exclusive*/>(temp, in2[x]);
      SumReduction<T>::template fold<true /*exclusive*/>(results[tid], temp);
    }
  }
  T result = results[0];
  for (int i = 1; i < max_threads; i++)
    SumReduction<T>::template fold<true /*exclusive*/>(result, results[i]);
  return result;
}
#endif

// We only have support for float and double versions of these currently
INSTANTIATE_REAL_TASKS(DotTask, static_cast<int>(NumPyOpCode::NUMPY_DOT) * NUMPY_TYPE_OFFSET)
// Full support for dot for all other types
INSTANTIATE_ALL_TASKS(DotReducTask, static_cast<int>(NumPyOpCode::NUMPY_DOT) * NUMPY_TYPE_OFFSET + NUMPY_REDUCTION_VARIANT_OFFSET)
}    // namespace numpy
}    // namespace legate

namespace    // unnammed
{
static void __attribute__((constructor)) register_tasks(void) {
  REGISTER_REAL_TASKS(legate::numpy::DotTask)
  REGISTER_ALL_TASKS_WITH_REDUCTION_RETURN(legate::numpy::DotReducTask, SumReduction)
}
}    // namespace
