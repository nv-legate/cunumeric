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
  CUNUMERIC_UNIQUE,
  CUNUMERIC_UNIQUE_REDUCE,
  CUNUMERIC_UNLOAD_CUDALIBS,
  CUNUMERIC_WHERE,
  CUNUMERIC_WRITE,
};

// Match these to UnaryOpCode in config.py
// Also, sort these alphabetically for easy lookup later
enum CuNumericUnaryOpCode {
  CUNUMERIC_UOP_ABSOLUTE = 1,
  CUNUMERIC_UOP_ARCCOS,
  CUNUMERIC_UOP_ARCSIN,
  CUNUMERIC_UOP_ARCTAN,
  CUNUMERIC_UOP_CEIL,
  CUNUMERIC_UOP_CLIP,
  CUNUMERIC_UOP_CONJ,
  CUNUMERIC_UOP_COPY,
  CUNUMERIC_UOP_COS,
  CUNUMERIC_UOP_EXP,
  CUNUMERIC_UOP_EXP2,
  CUNUMERIC_UOP_FLOOR,
  CUNUMERIC_UOP_GETARG,
  CUNUMERIC_UOP_IMAG,
  CUNUMERIC_UOP_INVERT,
  CUNUMERIC_UOP_ISINF,
  CUNUMERIC_UOP_ISNAN,
  CUNUMERIC_UOP_LOG,
  CUNUMERIC_UOP_LOG10,
  CUNUMERIC_UOP_LOGICAL_NOT,
  CUNUMERIC_UOP_NEGATIVE,
  CUNUMERIC_UOP_REAL,
  CUNUMERIC_UOP_RINT,
  CUNUMERIC_UOP_SIGN,
  CUNUMERIC_UOP_SIN,
  CUNUMERIC_UOP_SQRT,
  CUNUMERIC_UOP_TAN,
  CUNUMERIC_UOP_TANH,
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
