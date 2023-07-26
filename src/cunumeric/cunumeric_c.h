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

#ifndef LEGATE_USE_PYTHON_CFFI
#include "legate_preamble.h"
#include "core/legate_c.h"
#endif

// Match these to CuNumericOpCode in config.py
// Also, sort these alphabetically except the first one for easy lookup later
enum CuNumericOpCode {
  _CUNUMERIC_OP_CODE_BASE = 0,
  CUNUMERIC_ADVANCED_INDEXING,
  CUNUMERIC_ARANGE,
  CUNUMERIC_ARGWHERE,
  CUNUMERIC_BINARY_OP,
  CUNUMERIC_BINARY_RED,
  CUNUMERIC_BINCOUNT,
  CUNUMERIC_BITGENERATOR,
  CUNUMERIC_CHOOSE,
  CUNUMERIC_CONTRACT,
  CUNUMERIC_CONVERT,
  CUNUMERIC_CONVOLVE,
  CUNUMERIC_SCAN_GLOBAL,
  CUNUMERIC_SCAN_LOCAL,
  CUNUMERIC_DIAG,
  CUNUMERIC_DOT,
  CUNUMERIC_EYE,
  CUNUMERIC_FFT,
  CUNUMERIC_FILL,
  CUNUMERIC_FLIP,
  CUNUMERIC_GEMM,
  CUNUMERIC_HISTOGRAM,
  CUNUMERIC_LOAD_CUDALIBS,
  CUNUMERIC_MATMUL,
  CUNUMERIC_MATVECMUL,
  CUNUMERIC_NONZERO,
  CUNUMERIC_PACKBITS,
  CUNUMERIC_POTRF,
  CUNUMERIC_PUTMASK,
  CUNUMERIC_RAND,
  CUNUMERIC_READ,
  CUNUMERIC_REPEAT,
  CUNUMERIC_SCALAR_UNARY_RED,
  CUNUMERIC_SEARCHSORTED,
  CUNUMERIC_SOLVE,
  CUNUMERIC_SORT,
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
  CUNUMERIC_UNPACKBITS,
  CUNUMERIC_WHERE,
  CUNUMERIC_WINDOW,
  CUNUMERIC_WRAP,
  CUNUMERIC_WRITE,
  CUNUMERIC_ZIP,
};

// Match these to UnaryOpCode in config.py
// Also, sort these alphabetically for easy lookup later
enum CuNumericUnaryOpCode {
  CUNUMERIC_UOP_ABSOLUTE = 1,
  CUNUMERIC_UOP_ARCCOS,
  CUNUMERIC_UOP_ARCCOSH,
  CUNUMERIC_UOP_ARCSIN,
  CUNUMERIC_UOP_ARCSINH,
  CUNUMERIC_UOP_ARCTAN,
  CUNUMERIC_UOP_ARCTANH,
  CUNUMERIC_UOP_CBRT,
  CUNUMERIC_UOP_CEIL,
  CUNUMERIC_UOP_CLIP,
  CUNUMERIC_UOP_CONJ,
  CUNUMERIC_UOP_COPY,
  CUNUMERIC_UOP_COS,
  CUNUMERIC_UOP_COSH,
  CUNUMERIC_UOP_DEG2RAD,
  CUNUMERIC_UOP_EXP,
  CUNUMERIC_UOP_EXP2,
  CUNUMERIC_UOP_EXPM1,
  CUNUMERIC_UOP_FLOOR,
  CUNUMERIC_UOP_FREXP,
  CUNUMERIC_UOP_GETARG,
  CUNUMERIC_UOP_IMAG,
  CUNUMERIC_UOP_INVERT,
  CUNUMERIC_UOP_ISFINITE,
  CUNUMERIC_UOP_ISINF,
  CUNUMERIC_UOP_ISNAN,
  CUNUMERIC_UOP_LOG,
  CUNUMERIC_UOP_LOG10,
  CUNUMERIC_UOP_LOG1P,
  CUNUMERIC_UOP_LOG2,
  CUNUMERIC_UOP_LOGICAL_NOT,
  CUNUMERIC_UOP_MODF,
  CUNUMERIC_UOP_NEGATIVE,
  CUNUMERIC_UOP_POSITIVE,
  CUNUMERIC_UOP_RAD2DEG,
  CUNUMERIC_UOP_REAL,
  CUNUMERIC_UOP_RECIPROCAL,
  CUNUMERIC_UOP_RINT,
  CUNUMERIC_UOP_SIGN,
  CUNUMERIC_UOP_SIGNBIT,
  CUNUMERIC_UOP_SIN,
  CUNUMERIC_UOP_SINH,
  CUNUMERIC_UOP_SQRT,
  CUNUMERIC_UOP_SQUARE,
  CUNUMERIC_UOP_TAN,
  CUNUMERIC_UOP_TANH,
  CUNUMERIC_UOP_TRUNC,
};

// Match these to UnaryRedCode in config.py
// Also, sort these alphabetically for easy lookup later
enum CuNumericUnaryRedCode {
  CUNUMERIC_RED_ALL = 1,
  CUNUMERIC_RED_ANY,
  CUNUMERIC_RED_ARGMAX,
  CUNUMERIC_RED_ARGMIN,
  CUNUMERIC_RED_CONTAINS,
  CUNUMERIC_RED_COUNT_NONZERO,
  CUNUMERIC_RED_MAX,
  CUNUMERIC_RED_MIN,
  CUNUMERIC_RED_NANARGMAX,
  CUNUMERIC_RED_NANARGMIN,
  CUNUMERIC_RED_NANMAX,
  CUNUMERIC_RED_NANMIN,
  CUNUMERIC_RED_NANPROD,
  CUNUMERIC_RED_NANSUM,
  CUNUMERIC_RED_PROD,
  CUNUMERIC_RED_SUM,
};

// Match these to BinaryOpCode in config.py
// Also, sort these alphabetically for easy lookup later
enum CuNumericBinaryOpCode {
  CUNUMERIC_BINOP_ADD = 1,
  CUNUMERIC_BINOP_ARCTAN2,
  CUNUMERIC_BINOP_BITWISE_AND,
  CUNUMERIC_BINOP_BITWISE_OR,
  CUNUMERIC_BINOP_BITWISE_XOR,
  CUNUMERIC_BINOP_COPYSIGN,
  CUNUMERIC_BINOP_DIVIDE,
  CUNUMERIC_BINOP_EQUAL,
  CUNUMERIC_BINOP_FLOAT_POWER,
  CUNUMERIC_BINOP_FLOOR_DIVIDE,
  CUNUMERIC_BINOP_FMOD,
  CUNUMERIC_BINOP_GCD,
  CUNUMERIC_BINOP_GREATER,
  CUNUMERIC_BINOP_GREATER_EQUAL,
  CUNUMERIC_BINOP_HYPOT,
  CUNUMERIC_BINOP_ISCLOSE,
  CUNUMERIC_BINOP_LCM,
  CUNUMERIC_BINOP_LDEXP,
  CUNUMERIC_BINOP_LEFT_SHIFT,
  CUNUMERIC_BINOP_LESS,
  CUNUMERIC_BINOP_LESS_EQUAL,
  CUNUMERIC_BINOP_LOGADDEXP,
  CUNUMERIC_BINOP_LOGADDEXP2,
  CUNUMERIC_BINOP_LOGICAL_AND,
  CUNUMERIC_BINOP_LOGICAL_OR,
  CUNUMERIC_BINOP_LOGICAL_XOR,
  CUNUMERIC_BINOP_MAXIMUM,
  CUNUMERIC_BINOP_MINIMUM,
  CUNUMERIC_BINOP_MOD,
  CUNUMERIC_BINOP_MULTIPLY,
  CUNUMERIC_BINOP_NEXTAFTER,
  CUNUMERIC_BINOP_NOT_EQUAL,
  CUNUMERIC_BINOP_POWER,
  CUNUMERIC_BINOP_RIGHT_SHIFT,
  CUNUMERIC_BINOP_SUBTRACT,
};

// Match these to WindowOpCode in config.py
// Also, sort these alphabetically for easy lookup later
enum CuNumericWindowOpCode {
  CUNUMERIC_WINDOW_BARLETT = 1,
  CUNUMERIC_WINDOW_BLACKMAN,
  CUNUMERIC_WINDOW_HAMMING,
  CUNUMERIC_WINDOW_HANNING,
  CUNUMERIC_WINDOW_KAISER,
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
};

enum CuNumericBounds {
  CUNUMERIC_MAX_REDOPS = 1024,
  CUNUMERIC_MAX_TASKS  = 1048576,
};

// Match these to ScanCode in config.py
// Also, sort these alphabetically for easy lookup later
enum CuNumericScanCode {
  CUNUMERIC_SCAN_PROD = 1,
  CUNUMERIC_SCAN_SUM,
};

// Match these to ConvertCode in config.py
// Also, sort these alphabetically for easy lookup later
enum CuNumericConvertCode {
  CUNUMERIC_CONVERT_NAN_NOOP = 1,
  CUNUMERIC_CONVERT_NAN_PROD,
  CUNUMERIC_CONVERT_NAN_SUM,
};

// Match these to BitGeneratorOperation in config.py
enum CuNumericBitGeneratorOperation {
  CUNUMERIC_BITGENOP_CREATE       = 1,
  CUNUMERIC_BITGENOP_DESTROY      = 2,
  CUNUMERIC_BITGENOP_RAND_RAW     = 3,
  CUNUMERIC_BITGENOP_DISTRIBUTION = 4,
};

// Match these to BitGeneratorType in config.py
enum CuNumericBitGeneratorType {
  CUNUMERIC_BITGENTYPE_DEFAULT       = 0,
  CUNUMERIC_BITGENTYPE_XORWOW        = 1,
  CUNUMERIC_BITGENTYPE_MRG32K3A      = 2,
  CUNUMERIC_BITGENTYPE_MTGP32        = 3,
  CUNUMERIC_BITGENTYPE_MT19937       = 4,
  CUNUMERIC_BITGENTYPE_PHILOX4_32_10 = 5,
};

// Match these to BitGeneratorDistribution in config.py
enum CuNumericBitGeneratorDistribution {
  CUNUMERIC_BITGENDIST_INTEGERS_16 = 1,
  CUNUMERIC_BITGENDIST_INTEGERS_32,
  CUNUMERIC_BITGENDIST_INTEGERS_64,
  CUNUMERIC_BITGENDIST_UNIFORM_32,
  CUNUMERIC_BITGENDIST_UNIFORM_64,
  CUNUMERIC_BITGENDIST_LOGNORMAL_32,
  CUNUMERIC_BITGENDIST_LOGNORMAL_64,
  CUNUMERIC_BITGENDIST_NORMAL_32,
  CUNUMERIC_BITGENDIST_NORMAL_64,
  CUNUMERIC_BITGENDIST_POISSON,
  CUNUMERIC_BITGENDIST_EXPONENTIAL_32,
  CUNUMERIC_BITGENDIST_EXPONENTIAL_64,
  CUNUMERIC_BITGENDIST_GUMBEL_32,
  CUNUMERIC_BITGENDIST_GUMBEL_64,
  CUNUMERIC_BITGENDIST_LAPLACE_32,
  CUNUMERIC_BITGENDIST_LAPLACE_64,
  CUNUMERIC_BITGENDIST_LOGISTIC_32,
  CUNUMERIC_BITGENDIST_LOGISTIC_64,
  CUNUMERIC_BITGENDIST_PARETO_32,
  CUNUMERIC_BITGENDIST_PARETO_64,
  CUNUMERIC_BITGENDIST_POWER_32,
  CUNUMERIC_BITGENDIST_POWER_64,
  CUNUMERIC_BITGENDIST_RAYLEIGH_32,
  CUNUMERIC_BITGENDIST_RAYLEIGH_64,
  CUNUMERIC_BITGENDIST_CAUCHY_32,
  CUNUMERIC_BITGENDIST_CAUCHY_64,
  CUNUMERIC_BITGENDIST_TRIANGULAR_32,
  CUNUMERIC_BITGENDIST_TRIANGULAR_64,
  CUNUMERIC_BITGENDIST_WEIBULL_32,
  CUNUMERIC_BITGENDIST_WEIBULL_64,
  CUNUMERIC_BITGENDIST_BYTES,
  CUNUMERIC_BITGENDIST_BETA_32,
  CUNUMERIC_BITGENDIST_BETA_64,
  CUNUMERIC_BITGENDIST_F_32,
  CUNUMERIC_BITGENDIST_F_64,
  CUNUMERIC_BITGENDIST_LOGSERIES,
  CUNUMERIC_BITGENDIST_NONCENTRAL_F_32,
  CUNUMERIC_BITGENDIST_NONCENTRAL_F_64,
  CUNUMERIC_BITGENDIST_CHISQUARE_32,
  CUNUMERIC_BITGENDIST_CHISQUARE_64,
  CUNUMERIC_BITGENDIST_GAMMA_32,
  CUNUMERIC_BITGENDIST_GAMMA_64,
  CUNUMERIC_BITGENDIST_STANDARD_T_32,
  CUNUMERIC_BITGENDIST_STANDARD_T_64,
  CUNUMERIC_BITGENDIST_HYPERGEOMETRIC,
  CUNUMERIC_BITGENDIST_VONMISES_32,
  CUNUMERIC_BITGENDIST_VONMISES_64,
  CUNUMERIC_BITGENDIST_ZIPF,
  CUNUMERIC_BITGENDIST_GEOMETRIC,
  CUNUMERIC_BITGENDIST_WALD_32,
  CUNUMERIC_BITGENDIST_WALD_64,
  CUNUMERIC_BITGENDIST_BINOMIAL,
  CUNUMERIC_BITGENDIST_NEGATIVE_BINOMIAL,
};

// These fft types match CuNumericFFTType in config.py and cufftType
enum CuNumericFFTType {
  CUNUMERIC_FFT_R2C = 0x2a,  // Real to complex (interleaved)
  CUNUMERIC_FFT_C2R = 0x2c,  // Complex (interleaved) to real
  CUNUMERIC_FFT_C2C = 0x29,  // Complex to complex (interleaved)
  CUNUMERIC_FFT_D2Z = 0x6a,  // Double to double-complex (interleaved)
  CUNUMERIC_FFT_Z2D = 0x6c,  // Double-complex (interleaved) to double
  CUNUMERIC_FFT_Z2Z = 0x69   // Double-complex to double-complex (interleaved)
};

// These fft types match CuNumericFFTDirection in config.py and cufftDirection
enum CuNumericFFTDirection { CUNUMERIC_FFT_FORWARD = -1, CUNUMERIC_FFT_INVERSE = 1 };

// Match these to Bitorder in config.py
enum CuNumericBitorder { CUNUMERIC_BITORDER_BIG = 0, CUNUMERIC_BITORDER_LITTLE = 1 };

#ifdef __cplusplus
extern "C" {
#endif

void cunumeric_perform_registration();
bool cunumeric_has_curand();
void cunumeric_register_reduction_op(int32_t type_uid, int32_t elem_type_code);

#ifdef __cplusplus
}
#endif

#endif  // __CUNUMERIC_C_H__
