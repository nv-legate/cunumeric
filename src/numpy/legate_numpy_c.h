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
  NUMPY_ARANGE           = 1,
  NUMPY_BINARY_OP        = 2,
  NUMPY_BINARY_RED       = 3,
  NUMPY_BINCOUNT         = 4,
  NUMPY_CONVERT          = 5,
  NUMPY_DIAG             = 6,
  NUMPY_DOT              = 7,
  NUMPY_EYE              = 8,
  NUMPY_FILL             = 9,
  NUMPY_MATMUL           = 10,
  NUMPY_MATVECMUL        = 11,
  NUMPY_NONZERO          = 12,
  NUMPY_RAND             = 13,
  NUMPY_READ             = 14,
  NUMPY_SCALAR_UNARY_RED = 15,
  NUMPY_TILE             = 16,
  NUMPY_TRANSPOSE        = 17,
  NUMPY_UNARY_OP         = 18,
  NUMPY_UNARY_RED        = 19,
  NUMPY_WHERE            = 20,
  NUMPY_WRITE            = 21,
  NUMPY_CONVOLVE         = 22,
  NUMPY_FLIP             = 23,
};

// Match these to NumPyRedopCode in legate/numpy/config.py
enum NumPyRedopID {
  NUMPY_ARGMAX_REDOP = 1,
  NUMPY_ARGMIN_REDOP = 2,
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
