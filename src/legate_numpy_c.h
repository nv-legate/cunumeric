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

// Match these to NumPyVariantCode in legate/numpy/config.py
enum NumPyVariantCode {
  NUMPY_NORMAL_VARIANT_OFFSET            = 0,    // many_a x many_b -> many_c
  NUMPY_SCALAR_VARIANT_OFFSET            = 1,    // one_a x one_b -> one_c
  NUMPY_BROADCAST_VARIANT_OFFSET         = 2,    // one_a x many_b -> many_c
  NUMPY_REDUCTION_VARIANT_OFFSET         = 3,    // many_a x many_b -> one_c
  NUMPY_INPLACE_VARIANT_OFFSET           = 4,    // many_a x many_b -> many_a
  NUMPY_INPLACE_BROADCAST_VARIANT_OFFSET = 5,    // many_a x one_b -> many_a
  NUMPY_MAX_VARIANTS                     = 6     // this must be last
};

// Match these to NumPyOpCode in legate/numpy/config.py
enum NumPyOpCode {
  NUMPY_ABSOLUTE            = 0,
  NUMPY_ADD                 = 1,
  NUMPY_ALLCLOSE            = 2,
  NUMPY_ARCCOS              = 3,
  NUMPY_ARCSIN              = 4,
  NUMPY_ARCTAN              = 5,
  NUMPY_ARGMAX              = 6,
  NUMPY_ARGMAX_RADIX        = 7,
  NUMPY_ARGMIN              = 8,
  NUMPY_ARGMIN_RADIX        = 9,
  NUMPY_BINCOUNT            = 10,
  NUMPY_CEIL                = 11,
  NUMPY_CLIP                = 12,
  NUMPY_CONVERT             = 13,
  NUMPY_COPY                = 14,
  NUMPY_COS                 = 15,
  NUMPY_DIAG                = 16,
  NUMPY_DIVIDE              = 17,
  NUMPY_DOT                 = 18,
  NUMPY_EQUAL               = 19,
  NUMPY_EXP                 = 20,
  NUMPY_EYE                 = 21,
  NUMPY_FILL                = 22,
  NUMPY_FLOOR               = 23,
  NUMPY_FLOOR_DIVIDE        = 24,
  NUMPY_GETARG              = 25,
  NUMPY_GREATER             = 26,
  NUMPY_GREATER_EQUAL       = 27,
  NUMPY_INVERT              = 28,
  NUMPY_ISINF               = 29,
  NUMPY_ISNAN               = 30,
  NUMPY_LESS                = 31,
  NUMPY_LESS_EQUAL          = 32,
  NUMPY_LOG                 = 33,
  NUMPY_LOGICAL_NOT         = 34,
  NUMPY_MAX                 = 35,
  NUMPY_MAX_RADIX           = 36,
  NUMPY_MAXIMUM             = 37,
  NUMPY_MIN                 = 38,
  NUMPY_MIN_RADIX           = 39,
  NUMPY_MINIMUM             = 40,
  NUMPY_MOD                 = 41,
  NUMPY_MULTIPLY            = 42,
  NUMPY_NEGATIVE            = 43,
  NUMPY_NORM                = 44,
  NUMPY_NOT_EQUAL           = 45,
  NUMPY_POWER               = 46,
  NUMPY_PROD                = 47,
  NUMPY_PROD_RADIX          = 48,
  NUMPY_RAND_INTEGER        = 49,
  NUMPY_RAND_NORMAL         = 50,
  NUMPY_RAND_UNIFORM        = 51,
  NUMPY_READ                = 52,
  NUMPY_SIN                 = 53,
  NUMPY_SORT                = 54,
  NUMPY_SQRT                = 55,
  NUMPY_SUBTRACT            = 56,
  NUMPY_SUM                 = 57,
  NUMPY_SUM_RADIX           = 58,
  NUMPY_TAN                 = 59,
  NUMPY_TANH                = 60,
  NUMPY_TILE                = 61,
  NUMPY_TRANSPOSE           = 62,
  NUMPY_WHERE               = 63,
  NUMPY_WRITE               = 64,
  NUMPY_LOGICAL_AND         = 65,
  NUMPY_LOGICAL_OR          = 66,
  NUMPY_LOGICAL_XOR         = 67,
  NUMPY_CONTAINS            = 68,
  NUMPY_COUNT_NONZERO       = 69,
  NUMPY_NONZERO             = 70,
  NUMPY_COUNT_NONZERO_REDUC = 71,
  NUMPY_INCLUSIVE_SCAN      = 72,
  NUMPY_CONVERT_TO_RECT     = 73,
  NUMPY_ARANGE              = 74,
};

// Match these to NumPyRedopCode in legate/core/config.py
enum NumPyRedopID {
  NUMPY_ARGMIN_REDOP,
  NUMPY_ARGMAX_REDOP,
};

// We provide a global class of projection functions
// Match these to NumPyProjCode in legate/core/config.py
enum NumPyProjectionCode {
  // 2D reduction
  NUMPY_PROJ_2D_1D_X = 1,    // keep x
  NUMPY_PROJ_2D_1D_Y = 2,    // keep y
  // 2D broadcast
  NUMPY_PROJ_2D_2D_X0 = 3,    // x, broadcast 0
  NUMPY_PROJ_2D_2D_0X = 4,    // broadcast 0, x
  NUMPY_PROJ_2D_2D_0Y = 5,    // broadcast 0, y
  NUMPY_PROJ_2D_2D_Y0 = 6,    // y, broadcast 0
  // 2D promotion
  NUMPY_PROJ_1D_2D_X = 7,    // 1D point becomes (x, 0)
  NUMPY_PROJ_1D_2D_Y = 8,    // 1D point becomes (0, x)
  // 2D transpose
  NUMPY_PROJ_2D_2D_YX = 9,    // transpose (x,y) to (y,x)
  // 3D reduction
  NUMPY_PROJ_3D_2D_XY = 10,    // keep x and y
  NUMPY_PROJ_3D_2D_XZ = 11,    // keep x and z
  NUMPY_PROJ_3D_2D_YZ = 12,    // keep y and z
  NUMPY_PROJ_3D_1D_X  = 13,    // keep x
  NUMPY_PROJ_3D_1D_Y  = 14,    // keep y
  NUMPY_PROJ_3D_1D_Z  = 15,    // keep z
  // 3D broadcast
  NUMPY_PROJ_3D_3D_XY = 16,    // keep x and y, broadcast z
  NUMPY_PROJ_3D_3D_XZ = 17,    // keep x and z, broadcast y
  NUMPY_PROJ_3D_3D_YZ = 18,    // keep y and z, broadcast x
  NUMPY_PROJ_3D_3D_X  = 19,    // keep x, broadcast y and z
  NUMPY_PROJ_3D_3D_Y  = 20,    // keep y, broadcast x and z
  NUMPY_PROJ_3D_3D_Z  = 21,    // keep z, broadcast x and y
  NUMPY_PROJ_3D_2D_XB = 22,    // y becomes x, broadcast z as y
  NUMPY_PROJ_3D_2D_BY = 23,    // broadcast y as x, z becomes y
  // 3D promotion
  NUMPY_PROJ_2D_3D_XY = 24,    // 2D point becomes (x, y, 0)
  NUMPY_PROJ_2D_3D_XZ = 25,    // 2D point becomes (x, 0, y)
  NUMPY_PROJ_2D_3D_YZ = 26,    // 2D point becomes (0, x, y)
  NUMPY_PROJ_1D_3D_X  = 27,    // 1D point becomes (x, 0, 0)
  NUMPY_PROJ_1D_3D_Y  = 28,    // 1D point becomes (0, x, 0)
  NUMPY_PROJ_1D_3D_Z  = 29,    // 1D point becomes (0, 0, x)
  // Radix 2D
  NUMPY_PROJ_RADIX_2D_X_4_0 = 30,
  NUMPY_PROJ_RADIX_2D_X_4_1 = 31,
  NUMPY_PROJ_RADIX_2D_X_4_2 = 32,
  NUMPY_PROJ_RADIX_2D_X_4_3 = 33,
  NUMPY_PROJ_RADIX_2D_Y_4_0 = 34,
  NUMPY_PROJ_RADIX_2D_Y_4_1 = 35,
  NUMPY_PROJ_RADIX_2D_Y_4_2 = 36,
  NUMPY_PROJ_RADIX_2D_Y_4_3 = 37,
  // Radix 3D
  NUMPY_PROJ_RADIX_3D_X_4_0 = 38,
  NUMPY_PROJ_RADIX_3D_X_4_1 = 39,
  NUMPY_PROJ_RADIX_3D_X_4_2 = 40,
  NUMPY_PROJ_RADIX_3D_X_4_3 = 41,
  NUMPY_PROJ_RADIX_3D_Y_4_0 = 42,
  NUMPY_PROJ_RADIX_3D_Y_4_1 = 43,
  NUMPY_PROJ_RADIX_3D_Y_4_2 = 44,
  NUMPY_PROJ_RADIX_3D_Y_4_3 = 45,
  NUMPY_PROJ_RADIX_3D_Z_4_0 = 46,
  NUMPY_PROJ_RADIX_3D_Z_4_1 = 47,
  NUMPY_PROJ_RADIX_3D_Z_4_2 = 48,
  NUMPY_PROJ_RADIX_3D_Z_4_3 = 49,
  // Flattening
  NUMPY_PROJ_ND_1D_C_ORDER = 50,
  // Must always be last
  NUMPY_PROJ_LAST = 51,
};

// We provide a global class of sharding functions
enum NumPyShardingCode {
  NUMPY_SHARD_TILE_1D       = 1,
  NUMPY_SHARD_TILE_2D       = 2,
  NUMPY_SHARD_TILE_3D       = 3,
  NUMPY_SHARD_TILE_2D_YX    = 4,    // transpose
  NUMPY_SHARD_TILE_3D_2D_XY = 5,
  NUMPY_SHARD_TILE_3D_2D_XZ = 6,
  NUMPY_SHARD_TILE_3D_2D_YZ = 7,
  // 2D Radix sharding functions
  NUMPY_SHARD_RADIX_2D_X_0 = 8,    // never instantiated
  NUMPY_SHARD_RADIX_2D_X_1 = 9,
  NUMPY_SHARD_RADIX_2D_X_2 = 10,
  NUMPY_SHARD_RADIX_2D_X_3 = 11,
  NUMPY_SHARD_RADIX_2D_X_4 = 12,
  NUMPY_SHARD_RADIX_2D_X_5 = 13,
  NUMPY_SHARD_RADIX_2D_X_6 = 14,
  NUMPY_SHARD_RADIX_2D_X_7 = 15,
  NUMPY_SHARD_RADIX_2D_X_8 = 16,
  NUMPY_SHARD_RADIX_2D_Y_0 = 17,    // never instantiated
  NUMPY_SHARD_RADIX_2D_Y_1 = 18,
  NUMPY_SHARD_RADIX_2D_Y_2 = 19,
  NUMPY_SHARD_RADIX_2D_Y_3 = 20,
  NUMPY_SHARD_RADIX_2D_Y_4 = 21,
  NUMPY_SHARD_RADIX_2D_Y_5 = 22,
  NUMPY_SHARD_RADIX_2D_Y_6 = 23,
  NUMPY_SHARD_RADIX_2D_Y_7 = 24,
  NUMPY_SHARD_RADIX_2D_Y_8 = 25,
  // 3D Radix sharding functions
  NUMPY_SHARD_RADIX_3D_X_0 = 64,    // never instantiated
  NUMPY_SHARD_RADIX_3D_X_1 = 65,
  NUMPY_SHARD_RADIX_3D_X_2 = 66,
  NUMPY_SHARD_RADIX_3D_X_3 = 67,
  NUMPY_SHARD_RADIX_3D_X_4 = 68,
  NUMPY_SHARD_RADIX_3D_X_5 = 69,
  NUMPY_SHARD_RADIX_3D_X_6 = 70,
  NUMPY_SHARD_RADIX_3D_X_7 = 71,
  NUMPY_SHARD_RADIX_3D_X_8 = 72,
  NUMPY_SHARD_RADIX_3D_Y_0 = 73,    // never instantiated
  NUMPY_SHARD_RADIX_3D_Y_1 = 74,
  NUMPY_SHARD_RADIX_3D_Y_2 = 75,
  NUMPY_SHARD_RADIX_3D_Y_3 = 76,
  NUMPY_SHARD_RADIX_3D_Y_4 = 77,
  NUMPY_SHARD_RADIX_3D_Y_5 = 78,
  NUMPY_SHARD_RADIX_3D_Y_6 = 79,
  NUMPY_SHARD_RADIX_3D_Y_7 = 80,
  NUMPY_SHARD_RADIX_3D_Y_8 = 81,
  NUMPY_SHARD_RADIX_3D_Z_0 = 82,    // never instantiated
  NUMPY_SHARD_RADIX_3D_Z_1 = 83,
  NUMPY_SHARD_RADIX_3D_Z_2 = 84,
  NUMPY_SHARD_RADIX_3D_Z_3 = 85,
  NUMPY_SHARD_RADIX_3D_Z_4 = 86,
  NUMPY_SHARD_RADIX_3D_Z_5 = 87,
  NUMPY_SHARD_RADIX_3D_Z_6 = 88,
  NUMPY_SHARD_RADIX_3D_Z_7 = 89,
  NUMPY_SHARD_RADIX_3D_Z_8 = 90,
  NUMPY_SHARD_EXTRA        = 91,
  // Leave space for some extra IDs for transform sharding functions
  NUMPY_SHARD_LAST = 1024,    // This one must be last
};

enum NumpyTaskOffset {
  NUMPY_CONVERT_OFFSET  = 100000,
  NUMPY_BINCOUNT_OFFSET = 200000,
};

// Match these to NumPyMappingTag in legate/numpy/config.py
enum NumPyTag {
  NUMPY_SUBRANKABLE_TAG = 0x1,
  NUMPY_CPU_ONLY_TAG    = 0x2,
  NUMPY_GPU_ONLY_TAG    = 0x4,
  NUMPY_NO_MEMOIZE_TAG  = 0x8,
  NUMPY_KEY_REGION_TAG  = 0x10,
  NUMPY_RADIX_GEN_TAG   = 0xE0,     // Update radix gen shift if you change this
  NUMPY_RADIX_DIM_TAG   = 0x700,    // Update radix dim shift if you change this
};

// Match these to NumPyTunable in legate/numpy/config.py
enum NumPyTunable {
  NUMPY_TUNABLE_NUM_PIECES            = 1,
  NUMPY_TUNABLE_NUM_GPUS              = 2,
  NUMPY_TUNABLE_TOTAL_NODES           = 3,
  NUMPY_TUNABLE_LOCAL_CPUS            = 4,
  NUMPY_TUNABLE_LOCAL_GPUS            = 5,
  NUMPY_TUNABLE_LOCAL_OPENMPS         = 6,
  NUMPY_TUNABLE_RADIX                 = 7,
  NUMPY_TUNABLE_MIN_SHARD_VOLUME      = 8,
  NUMPY_TUNABLE_MAX_EAGER_VOLUME      = 9,
  NUMPY_TUNABLE_FIELD_REUSE_SIZE      = 10,
  NUMPY_TUNABLE_FIELD_REUSE_FREQUENCY = 11,
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

void legate_numpy_create_transform_sharding_functor(unsigned first, unsigned offset, unsigned M, unsigned N, const long* transform);

#ifdef __cplusplus
}
#endif

#endif    // __LEGATE_NUMPY_C_H__
