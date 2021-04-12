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

#include "trans.h"
#include "cblas.h"
#include "proj.h"

using namespace Legion;

namespace legate {
namespace numpy {

template<typename T, int BF>
static void trans_block_2d(const Task* task, Runtime* runtime, LegateDeserializer& derez,
                           const std::vector<PhysicalRegion>& regions) {
  const Rect<2> out_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
  if (out_rect.empty()) return;
  const AccessorWO<T, 2> out = derez.unpack_accessor_WO<T, 2>(regions[0], out_rect);
  // We know what the output shape has to be based on the input shape
  const Rect<2>          in_rect(Point<2>(out_rect.lo[1], out_rect.lo[0]), Point<2>(out_rect.hi[1], out_rect.hi[0]));
  const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], in_rect);
  for (coord_t i1 = in_rect.lo[0]; i1 <= in_rect.hi[0]; i1 += BF) {
    for (coord_t j1 = in_rect.lo[1]; j1 <= in_rect.hi[1]; j1 += BF) {
      const coord_t max_i2 = ((i1 + BF) <= in_rect.hi[0]) ? i1 + BF : in_rect.hi[0];
      const coord_t max_j2 = ((j1 + BF) <= in_rect.hi[1]) ? j1 + BF : in_rect.hi[1];
      for (int i2 = i1; i2 <= max_i2; i2++)
        for (int j2 = j1; j2 <= max_j2; j2++)
          out[j2][i2] = in[i2][j2];
    }
  }
}

#ifdef LEGATE_USE_OPENMP
template<typename T, int BF>
static void trans_block_2d_omp(const Task* task, Runtime* runtime, LegateDeserializer& derez,
                               const std::vector<PhysicalRegion>& regions) {
  const Rect<2> out_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
  if (out_rect.empty()) return;
  const AccessorWO<T, 2> out = derez.unpack_accessor_WO<T, 2>(regions[0], out_rect);
  // We know what the output shape has to be based on the input shape
  const Rect<2>          in_rect(Point<2>(out_rect.lo[1], out_rect.lo[0]), Point<2>(out_rect.hi[1], out_rect.hi[0]));
  const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[1], in_rect);
#  pragma omp            parallel for
  for (coord_t i1 = in_rect.lo[0]; i1 <= in_rect.hi[0]; i1 += BF) {
    for (coord_t j1 = in_rect.lo[1]; j1 <= in_rect.hi[1]; j1 += BF) {
      const coord_t max_i2 = ((i1 + BF) <= in_rect.hi[0]) ? i1 + BF : in_rect.hi[0];
      const coord_t max_j2 = ((j1 + BF) <= in_rect.hi[1]) ? j1 + BF : in_rect.hi[1];
      for (int i2 = i1; i2 <= max_i2; i2++)
        for (int j2 = j1; j2 <= max_j2; j2++)
          out[j2][i2] = in[i2][j2];
    }
  }
}
#endif

template<>
/*static*/ void TransTask<bool>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                             Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_block_2d<bool, 128>(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

template<>
/*static*/ void TransTask<int16_t>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_block_2d<int16_t, 64>(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

template<>
/*static*/ void TransTask<uint16_t>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                 Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_block_2d<uint16_t, 64>(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

template<>
/*static*/ void TransTask<__half>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                               Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_block_2d<__half, 64>(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

#ifdef LEGATE_USE_OPENMP
template<>
/*static*/ void TransTask<bool>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                             Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_block_2d_omp<bool, 128>(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

template<>
/*static*/ void TransTask<int16_t>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_block_2d_omp<int16_t, 64>(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

template<>
/*static*/ void TransTask<uint16_t>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                 Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_block_2d_omp<uint16_t, 64>(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

template<>
/*static*/ void TransTask<__half>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                               Runtime* runtime) {
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_block_2d_omp<__half, 64>(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}
#endif

static void trans_32bit_2d(const Task* task, Runtime* runtime, LegateDeserializer& derez,
                           const std::vector<PhysicalRegion>& regions) {
  const Rect<2> out_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
  if (out_rect.empty()) return;
  const AccessorWO<float, 2> out = derez.unpack_accessor_WO<float, 2>(regions[0], out_rect);
  size_t                     out_strides[2];
  float*                     out_ptr = out.ptr(out_rect, out_strides);
  // We know what the output shape has to be based on the input shape
  const Rect<2>              in_rect(Point<2>(out_rect.lo[1], out_rect.lo[0]), Point<2>(out_rect.hi[1], out_rect.hi[0]));
  const AccessorRO<float, 2> in = derez.unpack_accessor_RO<float, 2>(regions[1], in_rect);
  size_t                     in_strides[2];
  const float*               in_ptr = in.ptr(in_rect, in_strides);
  const coord_t              m      = (in_rect.hi[0] - in_rect.lo[0]) + 1;
  const coord_t              n      = (in_rect.hi[1] - in_rect.lo[1]) + 1;
  cblas_somatcopy(CblasRowMajor, CblasTrans, m, n, 1.f /*scale*/, in_ptr, in_strides[0], out_ptr, out_strides[0]);
}

template<>
/*static*/ void TransTask<float>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                              Runtime* runtime) {
  openblas_set_num_threads(1);    // make sure this isn't overzealous
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_32bit_2d(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

template<>
/*static*/ void TransTask<uint32_t>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                 Runtime* runtime) {
  openblas_set_num_threads(1);    // make sure this isn't overzealous
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_32bit_2d(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

template<>
/*static*/ void TransTask<int32_t>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                Runtime* runtime) {
  openblas_set_num_threads(1);    // make sure this isn't overzealous
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_32bit_2d(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

template<>
/*static*/ void TransTask<complex<__half>>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                        Runtime* runtime) {
  openblas_set_num_threads(1);    // make sure this isn't overzealous
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_32bit_2d(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

#ifdef LEGATE_USE_OPENMP
template<>
/*static*/ void TransTask<float>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                              Runtime* runtime) {
  // Assuming that openblas is built with OpenMP support we shouldn't need
  // to set the number of threads as it will use our OpenMP threads
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_32bit_2d(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

template<>
/*static*/ void TransTask<uint32_t>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                 Runtime* runtime) {
  // Assuming that openblas is built with OpenMP support we shouldn't need
  // to set the number of threads as it will use our OpenMP threads
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_32bit_2d(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

template<>
/*static*/ void TransTask<int32_t>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                Runtime* runtime) {
  // Assuming that openblas is built with OpenMP support we shouldn't need
  // to set the number of threads as it will use our OpenMP threads
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_32bit_2d(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

template<>
/*static*/ void TransTask<complex<__half>>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                        Runtime* runtime) {
  // Assuming that openblas is built with OpenMP support we shouldn't need
  // to set the number of threads as it will use our OpenMP threads
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_32bit_2d(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}
#endif

static void trans_64bit_2d(const Task* task, Runtime* runtime, LegateDeserializer& derez,
                           const std::vector<PhysicalRegion>& regions) {
  const Rect<2> out_rect = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
  if (out_rect.empty()) return;
  const AccessorWO<double, 2> out = derez.unpack_accessor_WO<double, 2>(regions[0], out_rect);
  size_t                      out_strides[2];
  double*                     out_ptr = out.ptr(out_rect, out_strides);
  // We know what the output shape has to be based on the input shape
  const Rect<2>               in_rect(Point<2>(out_rect.lo[1], out_rect.lo[0]), Point<2>(out_rect.hi[1], out_rect.hi[0]));
  const AccessorRO<double, 2> in = derez.unpack_accessor_RO<double, 2>(regions[1], in_rect);
  size_t                      in_strides[2];
  const double*               in_ptr = in.ptr(in_rect, in_strides);
  const coord_t               m      = (in_rect.hi[0] - in_rect.lo[0]) + 1;
  const coord_t               n      = (in_rect.hi[1] - in_rect.lo[1]) + 1;
  cblas_domatcopy(CblasRowMajor, CblasTrans, m, n, 1.0 /*scale*/, in_ptr, in_strides[0], out_ptr, out_strides[0]);
}

template<>
/*static*/ void TransTask<double>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                               Runtime* runtime) {
  openblas_set_num_threads(1);    // make sure this isn't overzealous
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_64bit_2d(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

template<>
/*static*/ void TransTask<int64_t>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                Runtime* runtime) {
  openblas_set_num_threads(1);    // make sure this isn't overzealous
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_64bit_2d(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

template<>
/*static*/ void TransTask<uint64_t>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                 Runtime* runtime) {
  openblas_set_num_threads(1);    // make sure this isn't overzealous
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_64bit_2d(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

template<>
/*static*/ void TransTask<complex<float>>::cpu_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                       Runtime* runtime) {
  openblas_set_num_threads(1);    // make sure this isn't overzealous
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_64bit_2d(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

#ifdef LEGATE_USE_OPENMP
template<>
/*static*/ void TransTask<double>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                               Runtime* runtime) {
  // Assuming that openblas is built with OpenMP support we shouldn't need
  // to set the number of threads as it will use our OpenMP threads
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_64bit_2d(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

template<>
/*static*/ void TransTask<int64_t>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                Runtime* runtime) {
  // Assuming that openblas is built with OpenMP support we shouldn't need
  // to set the number of threads as it will use our OpenMP threads
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_64bit_2d(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

template<>
/*static*/ void TransTask<uint64_t>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                 Runtime* runtime) {
  // Assuming that openblas is built with OpenMP support we shouldn't need
  // to set the number of threads as it will use our OpenMP threads
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_64bit_2d(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}

template<>
/*static*/ void TransTask<complex<float>>::omp_variant(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
                                                       Runtime* runtime) {
  // Assuming that openblas is built with OpenMP support we shouldn't need
  // to set the number of threads as it will use our OpenMP threads
  LegateDeserializer derez(task->args, task->arglen);
  const int          dim = derez.unpack_dimension();
  // We only support 2-D transposes for now
  switch (dim) {
    case 2: {
      trans_64bit_2d(task, runtime, derez, regions);
      break;
    }
    default:
      assert(false);    // unsupported transpose dimension
  }
}
#endif

INSTANTIATE_ALL_TASKS(TransTask, static_cast<int>(NumPyOpCode::NUMPY_TRANSPOSE) * NUMPY_TYPE_OFFSET)

}    // namespace numpy
}    // namespace legate

namespace    // unnammed
{
static void __attribute__((constructor)) register_tasks(void) { REGISTER_ALL_TASKS(legate::numpy::TransTask) }
}    // namespace
