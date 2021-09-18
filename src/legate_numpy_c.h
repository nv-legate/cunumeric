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

#ifndef __LEGATE_NUMPY_C_H__
#define __LEGATE_NUMPY_C_H__

#include "legate_preamble.h"

// Match these to NumPyOpCode in legate/numpy/config.py
enum NumPyOpCode {
  NUMPY_BINARY_OP        = 400000,
  NUMPY_SCALAR_BINARY_OP = 400002,
  NUMPY_FILL             = 400003,
  NUMPY_SCALAR_UNARY_RED = 400004,
  NUMPY_UNARY_RED        = 400005,
  NUMPY_UNARY_OP         = 400006,
  NUMPY_SCALAR_UNARY_OP  = 400007,
  NUMPY_BINARY_RED       = 400008,
  NUMPY_CONVERT          = 400010,
  NUMPY_SCALAR_CONVERT   = 400011,
  NUMPY_WHERE            = 400012,
  NUMPY_SCALAR_WHERE     = 400013,
  NUMPY_READ             = 400014,
  NUMPY_WRITE            = 400015,
  NUMPY_DIAG             = 400016,
  NUMPY_MATMUL           = 400017,
  NUMPY_MATVECMUL        = 400018,
  NUMPY_DOT              = 400019,
  NUMPY_BINCOUNT         = 400020,
  NUMPY_EYE              = 400021,
  NUMPY_RAND             = 400022,
  NUMPY_ARANGE           = 400023,
  NUMPY_TRANSPOSE        = 400024,
  NUMPY_TILE             = 400025,
  NUMPY_NONZERO          = 400026,
  NUMPY_FUSED_OP         = 400027,
};

// Match these to NumPyRedopCode in legate/numpy/config.py
enum NumPyRedopID {
  NUMPY_ARGMIN_REDOP = LEGION_REDOP_KIND_TOTAL + 1,
  NUMPY_ARGMAX_REDOP,
  NUMPY_SCALAR_MAX_REDOP    = 500,
  NUMPY_SCALAR_MIN_REDOP    = 501,
  NUMPY_SCALAR_PROD_REDOP   = 502,
  NUMPY_SCALAR_SUM_REDOP    = 503,
  NUMPY_SCALAR_ARGMAX_REDOP = 504,
  NUMPY_SCALAR_ARGMIN_REDOP = 505,
};

// Match these to NumPyMappingTag in legate/numpy/config.py
enum NumPyTag {
  NUMPY_SUBRANKABLE_TAG = 0x1,
  NUMPY_CPU_ONLY_TAG    = 0x2,
  NUMPY_GPU_ONLY_TAG    = 0x4,
  NUMPY_NO_MEMOIZE_TAG  = 0x8,
  NUMPY_KEY_REGION_TAG  = 0x10,
};

// Match these to NumPyTunable in legate/numpy/config.py
enum NumPyTunable {
  NUMPY_TUNABLE_NUM_GPUS         = 1,
  NUMPY_TUNABLE_MAX_EAGER_VOLUME = 2,
};

enum NumPyBounds {
  NUMPY_MAX_MAPPERS = 1,
  NUMPY_MAX_REDOPS  = 1024,
  NUMPY_MAX_TASKS   = 1048576,
};

#ifdef __cplusplus
extern "C" {
#endif

void legate_numpy_perform_registration();

#ifdef __cplusplus
}
#endif

#endif  // __LEGATE_NUMPY_C_H__
