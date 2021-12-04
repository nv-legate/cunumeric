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

#ifndef __CUNUMERIC_C_H__
#define __CUNUMERIC_C_H__

#include "legate_preamble.h"

// Match these to CuNumericOpCode in cunumeric/config.py
enum CuNumericOpCode {
  CUNUMERIC_ARANGE           = 1,
  CUNUMERIC_BINARY_OP        = 2,
  CUNUMERIC_BINARY_RED       = 3,
  CUNUMERIC_BINCOUNT         = 4,
  CUNUMERIC_CONVERT          = 5,
  CUNUMERIC_CONVOLVE         = 6,
  CUNUMERIC_DIAG             = 7,
  CUNUMERIC_DOT              = 8,
  CUNUMERIC_EYE              = 9,
  CUNUMERIC_FILL             = 10,
  CUNUMERIC_FLIP             = 11,
  CUNUMERIC_GEMM             = 12,
  CUNUMERIC_MATMUL           = 13,
  CUNUMERIC_MATVECMUL        = 14,
  CUNUMERIC_NONZERO          = 15,
  CUNUMERIC_POTRF            = 16,
  CUNUMERIC_RAND             = 17,
  CUNUMERIC_READ             = 18,
  CUNUMERIC_SCALAR_UNARY_RED = 19,
  CUNUMERIC_SYRK             = 20,
  CUNUMERIC_TILE             = 21,
  CUNUMERIC_TRANSPOSE        = 22,
  CUNUMERIC_TRSM             = 23,
  CUNUMERIC_UNARY_OP         = 24,
  CUNUMERIC_UNARY_RED        = 25,
  CUNUMERIC_WHERE            = 26,
  CUNUMERIC_WRITE            = 27,
};

// Match these to CuNumericRedopCode in cunumeric/config.py
enum CuNumericRedopID {
  CUNUMERIC_ARGMAX_REDOP = 1,
  CUNUMERIC_ARGMIN_REDOP = 2,
};

// Match these to CuNumericTunable in cunumeric/config.py
enum CuNumericTunable {
  CUNUMERIC_TUNABLE_NUM_GPUS         = 1,
  CUNUMERIC_TUNABLE_NUM_PROCS        = 2,
  CUNUMERIC_TUNABLE_MAX_EAGER_VOLUME = 3,
};

enum CuNumericBounds {
  CUNUMERIC_MAX_MAPPERS = 1,
  CUNUMERIC_MAX_REDOPS  = 1024,
  CUNUMERIC_MAX_TASKS   = 1048576,
};

#ifdef __cplusplus
extern "C" {
#endif

void cunumeric_perform_registration();

#ifdef __cplusplus
}
#endif

#endif  // __CUNUMERIC_C_H__
