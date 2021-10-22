# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See the LICENSE file for details.
#

# IMPORTANT:
#   * header.py.in is used as an input to string.format()
#   * header.py is a generated file and should not be modified by hand

header = 'enum CuNumericOpCode {\n  CUNUMERIC_ARANGE = 1,\n  CUNUMERIC_BINARY_OP = 2,\n  CUNUMERIC_BINARY_RED = 3,\n  CUNUMERIC_BINCOUNT = 4,\n  CUNUMERIC_CONVERT = 5,\n  CUNUMERIC_DIAG = 6,\n  CUNUMERIC_DOT = 7,\n  CUNUMERIC_EYE = 8,\n  CUNUMERIC_FILL = 9,\n  CUNUMERIC_MATMUL = 10,\n  CUNUMERIC_MATVECMUL = 11,\n  CUNUMERIC_NONZERO = 12,\n  CUNUMERIC_RAND = 13,\n  CUNUMERIC_READ = 14,\n  CUNUMERIC_SCALAR_UNARY_RED = 15,\n  CUNUMERIC_TILE = 16,\n  CUNUMERIC_TRANSPOSE = 17,\n  CUNUMERIC_UNARY_OP = 18,\n  CUNUMERIC_UNARY_RED = 19,\n  CUNUMERIC_WHERE = 20,\n  CUNUMERIC_WRITE = 21,\n};\nenum CuNumericRedopID {\n  CUNUMERIC_ARGMAX_REDOP = 1,\n  CUNUMERIC_ARGMIN_REDOP = 2,\n};\nenum CuNumericTunable {\n  CUNUMERIC_TUNABLE_NUM_GPUS = 1,\n  CUNUMERIC_TUNABLE_MAX_EAGER_VOLUME = 2,\n};\nenum CuNumericBounds {\n  CUNUMERIC_MAX_MAPPERS = 1,\n  CUNUMERIC_MAX_REDOPS = 1024,\n  CUNUMERIC_MAX_TASKS = 1048576,\n};\nvoid cunumeric_perform_registration();\n'
libpath = '/raid/wonchanl/Workspace/legate/lib'
