/* Copyright 2021-2022 NVIDIA Corporation
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

// Match these to CuNumericOpCode in config.py
// Also, sort these alphabetically except the first one for easy lookup later
enum CuNumericOpCode {
  _CUNUMERIC_OP_CODE_BASE = 0,
  CUNUMERIC_ARANGE,
  CUNUMERIC_BINARY_OP,
  CUNUMERIC_BINARY_RED,
  CUNUMERIC_BINCOUNT,
  CUNUMERIC_CHOOSE,
  CUNUMERIC_CONTRACT,
  CUNUMERIC_CONVERT,
  CUNUMERIC_CONVOLVE,
  CUNUMERIC_DIAG,
  CUNUMERIC_DOT,
  CUNUMERIC_EYE,
  CUNUMERIC_FILL,
  CUNUMERIC_FLIP,
  CUNUMERIC_GEMM,
  CUNUMERIC_LOAD_CUDALIBS,
  CUNUMERIC_MATMUL,
  CUNUMERIC_MATVECMUL,
  CUNUMERIC_NONZERO,
  CUNUMERIC_POTRF,
  CUNUMERIC_RAND,
  CUNUMERIC_READ,
  CUNUMERIC_SCALAR_UNARY_RED,
  CUNUMERIC_SYRK,
  CUNUMERIC_TILE,
  CUNUMERIC_TRANSPOSE_COPY_2D,
  CUNUMERIC_TRILU,
  CUNUMERIC_TRSM,
  CUNUMERIC_UNARY_OP,
  CUNUMERIC_UNARY_RED,
  CUNUMERIC_UNLOAD_CUDALIBS,
  CUNUMERIC_WHERE,
  CUNUMERIC_WRITE,
};

// Match these to CuNumericRedopCode in config.py
enum CuNumericRedopID {
  CUNUMERIC_ARGMAX_REDOP = 1,
  CUNUMERIC_ARGMIN_REDOP = 2,
};

// Match these to CuNumericTunable in config.py
enum CuNumericTunable {
  CUNUMERIC_TUNABLE_NUM_GPUS         = 1,
  CUNUMERIC_TUNABLE_NUM_PROCS        = 2,
  CUNUMERIC_TUNABLE_MAX_EAGER_VOLUME = 3,
  CUNUMERIC_TUNABLE_HAS_NUMAMEM      = 4,
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
