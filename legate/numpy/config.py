# Copyright 2021 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import absolute_import, division, print_function

import os
import sys
from enum import IntEnum, unique

import numpy as np

from legate.core import LegateLibrary, legate_add_library, legion


# Helper method for python 3 support
def _itervalues(obj):
    return obj.values() if sys.version_info > (3,) else obj.viewvalues()


# Load the Legate NumPy library first so we have a shard object that
# we can use to initialize all these configuration enumerations
class NumPyLib(LegateLibrary):
    def __init__(self, name):
        self.name = name
        self.runtime = None

    def get_name(self):
        return self.name

    def get_shared_library(self):
        from legate.numpy.install_info import libpath

        return os.path.join(
            libpath, "liblgnumpy" + self.get_library_extension()
        )

    def get_c_header(self):
        from legate.numpy.install_info import header

        return header

    def get_registration_callback(self):
        return "legate_numpy_perform_registration"

    def initialize(self, shared_object):
        assert self.runtime is None
        self.shared_object = shared_object

    def set_runtime(self, runtime):
        assert self.runtime is None
        assert self.shared_object is not None
        self.runtime = runtime

    def destroy(self):
        if self.runtime is not None:
            self.runtime.destroy()


NUMPY_LIB_NAME = "legate.numpy"
numpy_lib = NumPyLib(NUMPY_LIB_NAME)
legate_add_library(numpy_lib)
legate_numpy = numpy_lib.shared_object


# Match these to legate_core_type_code_t in legate_c.h
numpy_field_type_offsets = {
    np.bool_: legion.LEGION_TYPE_BOOL,
    np.int8: legion.LEGION_TYPE_INT8,
    np.int16: legion.LEGION_TYPE_INT16,
    np.int: legion.LEGION_TYPE_INT32,
    np.int32: legion.LEGION_TYPE_INT32,
    np.int64: legion.LEGION_TYPE_INT64,
    np.uint8: legion.LEGION_TYPE_UINT8,
    np.uint16: legion.LEGION_TYPE_UINT16,
    np.uint32: legion.LEGION_TYPE_UINT32,
    np.uint64: legion.LEGION_TYPE_UINT64,
    np.float16: legion.LEGION_TYPE_FLOAT16,
    np.float: legion.LEGION_TYPE_FLOAT32,
    np.float32: legion.LEGION_TYPE_FLOAT32,
    np.float64: legion.LEGION_TYPE_FLOAT64,
    np.complex64: legion.LEGION_TYPE_COMPLEX64,
    np.complex128: legion.LEGION_TYPE_COMPLEX128,
}


# Match these to NumPyVariantCode in legate_numpy_c.h
@unique
class NumPyVariantCode(IntEnum):
    NORMAL = legate_numpy.NUMPY_NORMAL_VARIANT_OFFSET
    SCALAR = legate_numpy.NUMPY_SCALAR_VARIANT_OFFSET
    BROADCAST = legate_numpy.NUMPY_BROADCAST_VARIANT_OFFSET
    REDUCTION = legate_numpy.NUMPY_REDUCTION_VARIANT_OFFSET
    INPLACE = legate_numpy.NUMPY_INPLACE_VARIANT_OFFSET
    INPLACE_BROADCAST = legate_numpy.NUMPY_INPLACE_BROADCAST_VARIANT_OFFSET


NUMPY_MAX_VARIANTS = len(NumPyVariantCode)
NUMPY_MAX_TYPES = legion.MAX_TYPE_NUMBER
NUMPY_TYPE_OFFSET = NUMPY_MAX_TYPES * NUMPY_MAX_VARIANTS


# Match these to NumPyOpCode in legate_numpy_c.h
@unique
class NumPyOpCode(IntEnum):
    ABSOLUTE = legate_numpy.NUMPY_ABSOLUTE
    ALLCLOSE = legate_numpy.NUMPY_ALLCLOSE
    ARCCOS = legate_numpy.NUMPY_ARCCOS
    ARCSIN = legate_numpy.NUMPY_ARCSIN
    ARCTAN = legate_numpy.NUMPY_ARCTAN
    ARGMAX = legate_numpy.NUMPY_ARGMAX
    ARGMAX_RADIX = legate_numpy.NUMPY_ARGMAX_RADIX
    ARGMIN = legate_numpy.NUMPY_ARGMIN
    ARGMIN_RADIX = legate_numpy.NUMPY_ARGMIN_RADIX
    BINCOUNT = legate_numpy.NUMPY_BINCOUNT
    CEIL = legate_numpy.NUMPY_CEIL
    CLIP = legate_numpy.NUMPY_CLIP
    CONVERT = legate_numpy.NUMPY_CONVERT
    COPY = legate_numpy.NUMPY_COPY
    COS = legate_numpy.NUMPY_COS
    DIAG = legate_numpy.NUMPY_DIAG
    DOT = legate_numpy.NUMPY_DOT
    EQUAL = legate_numpy.NUMPY_EQUAL
    EXP = legate_numpy.NUMPY_EXP
    EYE = legate_numpy.NUMPY_EYE
    FILL = legate_numpy.NUMPY_FILL
    FLOOR = legate_numpy.NUMPY_FLOOR
    GETARG = legate_numpy.NUMPY_GETARG
    GREATER = legate_numpy.NUMPY_GREATER
    GREATER_EQUAL = legate_numpy.NUMPY_GREATER_EQUAL
    INVERT = legate_numpy.NUMPY_INVERT
    ISINF = legate_numpy.NUMPY_ISINF
    ISNAN = legate_numpy.NUMPY_ISNAN
    LESS = legate_numpy.NUMPY_LESS
    LESS_EQUAL = legate_numpy.NUMPY_LESS_EQUAL
    LOG = legate_numpy.NUMPY_LOG
    LOGICAL_NOT = legate_numpy.NUMPY_LOGICAL_NOT
    MAX = legate_numpy.NUMPY_MAX
    MAX_RADIX = legate_numpy.NUMPY_MAX_RADIX
    MIN = legate_numpy.NUMPY_MIN
    MIN_RADIX = legate_numpy.NUMPY_MIN_RADIX
    MOD = legate_numpy.NUMPY_MOD
    NEGATIVE = legate_numpy.NUMPY_NEGATIVE
    NORM = legate_numpy.NUMPY_NORM
    NOT_EQUAL = legate_numpy.NUMPY_NOT_EQUAL
    PROD = legate_numpy.NUMPY_PROD
    PROD_RADIX = legate_numpy.NUMPY_PROD_RADIX
    RAND_INTEGER = legate_numpy.NUMPY_RAND_INTEGER
    RAND_NORMAL = legate_numpy.NUMPY_RAND_NORMAL
    RAND_UNIFORM = legate_numpy.NUMPY_RAND_UNIFORM
    READ = legate_numpy.NUMPY_READ
    SIN = legate_numpy.NUMPY_SIN
    SORT = legate_numpy.NUMPY_SORT
    SQRT = legate_numpy.NUMPY_SQRT
    SUM = legate_numpy.NUMPY_SUM
    SUM_RADIX = legate_numpy.NUMPY_SUM_RADIX
    TAN = legate_numpy.NUMPY_TAN
    TANH = legate_numpy.NUMPY_TANH
    TILE = legate_numpy.NUMPY_TILE
    TRANSPOSE = legate_numpy.NUMPY_TRANSPOSE
    WHERE = legate_numpy.NUMPY_WHERE
    WRITE = legate_numpy.NUMPY_WRITE
    CONTAINS = legate_numpy.NUMPY_CONTAINS
    COUNT_NONZERO = legate_numpy.NUMPY_COUNT_NONZERO
    NONZERO = legate_numpy.NUMPY_NONZERO
    COUNT_NONZERO_REDUC = legate_numpy.NUMPY_COUNT_NONZERO_REDUC
    INCLUSIVE_SCAN = legate_numpy.NUMPY_INCLUSIVE_SCAN
    CONVERT_TO_RECT = legate_numpy.NUMPY_CONVERT_TO_RECT
    ARANGE = legate_numpy.NUMPY_ARANGE
    BINARY_OP = legate_numpy.NUMPY_BINARY_OP
    BROADCAST_BINARY_OP = legate_numpy.NUMPY_BROADCAST_BINARY_OP
    SCALAR_BINARY_OP = legate_numpy.NUMPY_SCALAR_BINARY_OP


@unique
class BinaryOpCode(IntEnum):
    ADD = 1
    DIVIDE = 2
    EQUAL = 3
    FLOOR_DIVIDE = 4
    GREATER = 5
    GREATER_EQUAL = 6
    LESS = 7
    LESS_EQUAL = 8
    MAXIMUM = 9
    MINIMUM = 10
    MOD = 11
    MULTIPLY = 12
    NOT_EQUAL = 13
    POWER = 14
    SUBTRACT = 15


# Match these to NumPyRedopID in legate_numpy_c.h
@unique
class NumPyRedopCode(IntEnum):
    ARGMIN_REDOP = legate_numpy.NUMPY_ARGMIN_REDOP
    ARGMAX_REDOP = legate_numpy.NUMPY_ARGMAX_REDOP


numpy_reduction_op_offsets = {
    NumPyOpCode.SUM: legion.LEGION_REDOP_KIND_SUM,
    NumPyOpCode.PROD: legion.LEGION_REDOP_KIND_PROD,
    NumPyOpCode.MIN: legion.LEGION_REDOP_KIND_MIN,
    NumPyOpCode.MAX: legion.LEGION_REDOP_KIND_MAX,
    # Dot uses sum reduction
    NumPyOpCode.DOT: legion.LEGION_REDOP_KIND_SUM,
    # Diag uses sum reduction
    NumPyOpCode.DIAG: legion.LEGION_REDOP_KIND_SUM,
    NumPyOpCode.EQUAL: legion.LEGION_REDOP_KIND_PROD,
    NumPyOpCode.NOT_EQUAL: legion.LEGION_REDOP_KIND_PROD,
    NumPyOpCode.GREATER: legion.LEGION_REDOP_KIND_PROD,
    NumPyOpCode.GREATER_EQUAL: legion.LEGION_REDOP_KIND_PROD,
    NumPyOpCode.LESS: legion.LEGION_REDOP_KIND_PROD,
    NumPyOpCode.LESS_EQUAL: legion.LEGION_REDOP_KIND_PROD,
    NumPyOpCode.ALLCLOSE: legion.LEGION_REDOP_KIND_PROD,
    # Norm uses sum reduction
    NumPyOpCode.NORM: legion.LEGION_REDOP_KIND_SUM,
    NumPyOpCode.ARGMIN: NumPyRedopCode.ARGMIN_REDOP,
    NumPyOpCode.ARGMAX: NumPyRedopCode.ARGMAX_REDOP,
    # bool sum is "or"
    NumPyOpCode.CONTAINS: legion.LEGION_REDOP_KIND_SUM,
    # nonzeros are counted with sum
    NumPyOpCode.COUNT_NONZERO: legion.LEGION_REDOP_KIND_SUM,
}


# Match these to NumPyTunable in legate_numpy_c.h
@unique
class NumPyTunable(IntEnum):
    NUM_PIECES = legate_numpy.NUMPY_TUNABLE_NUM_PIECES
    NUM_GPUS = legate_numpy.NUMPY_TUNABLE_NUM_GPUS
    TOTAL_NODES = legate_numpy.NUMPY_TUNABLE_TOTAL_NODES
    LOCAL_CPUS = legate_numpy.NUMPY_TUNABLE_LOCAL_CPUS
    LOCAL_GPUS = legate_numpy.NUMPY_TUNABLE_LOCAL_GPUS
    LOCAL_OMPS = legate_numpy.NUMPY_TUNABLE_LOCAL_OPENMPS
    RADIX = legate_numpy.NUMPY_TUNABLE_RADIX
    MIN_SHARD_VOLUME = legate_numpy.NUMPY_TUNABLE_MIN_SHARD_VOLUME
    MAX_EAGER_VOLUME = legate_numpy.NUMPY_TUNABLE_MAX_EAGER_VOLUME
    FIELD_REUSE_SIZE = legate_numpy.NUMPY_TUNABLE_FIELD_REUSE_SIZE
    FIELD_REUSE_FREQ = legate_numpy.NUMPY_TUNABLE_FIELD_REUSE_FREQUENCY


# Match these to NumPyTag in legate_numpy_c.h
@unique
class NumPyMappingTag(IntEnum):
    SUBRANKABLE_TASK_TAG = legate_numpy.NUMPY_SUBRANKABLE_TAG
    CPU_ONLY_TASK_TAG = legate_numpy.NUMPY_CPU_ONLY_TAG
    GPU_ONLY_TASK_TAG = legate_numpy.NUMPY_GPU_ONLY_TAG
    NO_MEMOIZE_TAG = 0  # Turn this off for now since it doesn't help
    KEY_REGION_TAG = legate_numpy.NUMPY_KEY_REGION_TAG
    RADIX_GEN_TAG = legate_numpy.NUMPY_RADIX_GEN_TAG
    RADIX_DIM_TAG = legate_numpy.NUMPY_RADIX_DIM_TAG


RADIX_GEN_SHIFT = 5
RADIX_DIM_SHIFT = 8


# Match these to NumPyProjectionCode in legate_numpy_c.h
@unique
class NumPyProjCode(IntEnum):
    # 2D reduction
    PROJ_2D_1D_X = legate_numpy.NUMPY_PROJ_2D_1D_X
    PROJ_2D_1D_Y = legate_numpy.NUMPY_PROJ_2D_1D_Y
    # 2D broadcast
    PROJ_2D_2D_X = legate_numpy.NUMPY_PROJ_2D_2D_X
    PROJ_2D_2D_Y = legate_numpy.NUMPY_PROJ_2D_2D_Y
    # 2D promotion
    PROJ_1D_2D_X = legate_numpy.NUMPY_PROJ_1D_2D_X
    PROJ_1D_2D_Y = legate_numpy.NUMPY_PROJ_1D_2D_Y
    # 2D transpose
    PROJ_2D_2D_YX = legate_numpy.NUMPY_PROJ_2D_2D_YX
    # 3D reduction
    PROJ_3D_2D_XY = legate_numpy.NUMPY_PROJ_3D_2D_XY
    PROJ_3D_2D_XZ = legate_numpy.NUMPY_PROJ_3D_2D_XZ
    PROJ_3D_2D_YZ = legate_numpy.NUMPY_PROJ_3D_2D_YZ
    PROJ_3D_1D_X = legate_numpy.NUMPY_PROJ_3D_1D_X
    PROJ_3D_1D_Y = legate_numpy.NUMPY_PROJ_3D_1D_Y
    PROJ_3D_1D_Z = legate_numpy.NUMPY_PROJ_3D_1D_Z
    # 3D broadcast
    PROJ_3D_3D_XY = legate_numpy.NUMPY_PROJ_3D_3D_XY
    PROJ_3D_3D_XZ = legate_numpy.NUMPY_PROJ_3D_3D_XZ
    PROJ_3D_3D_YZ = legate_numpy.NUMPY_PROJ_3D_3D_YZ
    PROJ_3D_3D_X = legate_numpy.NUMPY_PROJ_3D_3D_X
    PROJ_3D_3D_Y = legate_numpy.NUMPY_PROJ_3D_3D_Y
    PROJ_3D_3D_Z = legate_numpy.NUMPY_PROJ_3D_3D_Z
    PROJ_3D_2D_XB = legate_numpy.NUMPY_PROJ_3D_2D_XB
    PROJ_3D_2D_BY = legate_numpy.NUMPY_PROJ_3D_2D_BY
    # 3D promotion
    PROJ_2D_3D_XY = legate_numpy.NUMPY_PROJ_2D_3D_XY
    PROJ_2D_3D_XZ = legate_numpy.NUMPY_PROJ_2D_3D_XZ
    PROJ_2D_3D_YZ = legate_numpy.NUMPY_PROJ_2D_3D_YZ
    PROJ_1D_3D_X = legate_numpy.NUMPY_PROJ_1D_3D_X
    PROJ_1D_3D_Y = legate_numpy.NUMPY_PROJ_1D_3D_Y
    PROJ_1D_3D_Z = legate_numpy.NUMPY_PROJ_1D_3D_Z
    # Radix 2D
    PROJ_RADIX_2D_X_4_0 = legate_numpy.NUMPY_PROJ_RADIX_2D_X_4_0
    PROJ_RADIX_2D_X_4_1 = legate_numpy.NUMPY_PROJ_RADIX_2D_X_4_1
    PROJ_RADIX_2D_X_4_2 = legate_numpy.NUMPY_PROJ_RADIX_2D_X_4_2
    PROJ_RADIX_2D_X_4_3 = legate_numpy.NUMPY_PROJ_RADIX_2D_X_4_3
    PROJ_RADIX_2D_Y_4_0 = legate_numpy.NUMPY_PROJ_RADIX_2D_Y_4_0
    PROJ_RADIX_2D_Y_4_1 = legate_numpy.NUMPY_PROJ_RADIX_2D_Y_4_1
    PROJ_RADIX_2D_Y_4_2 = legate_numpy.NUMPY_PROJ_RADIX_2D_Y_4_2
    PROJ_RADIX_2D_Y_4_3 = legate_numpy.NUMPY_PROJ_RADIX_2D_Y_4_3
    # Radix 3D
    PROJ_RADIX_3D_X_4_0 = legate_numpy.NUMPY_PROJ_RADIX_3D_X_4_0
    PROJ_RADIX_3D_X_4_1 = legate_numpy.NUMPY_PROJ_RADIX_3D_X_4_1
    PROJ_RADIX_3D_X_4_2 = legate_numpy.NUMPY_PROJ_RADIX_3D_X_4_2
    PROJ_RADIX_3D_X_4_3 = legate_numpy.NUMPY_PROJ_RADIX_3D_X_4_3
    PROJ_RADIX_3D_Y_4_0 = legate_numpy.NUMPY_PROJ_RADIX_3D_Y_4_0
    PROJ_RADIX_3D_Y_4_1 = legate_numpy.NUMPY_PROJ_RADIX_3D_Y_4_1
    PROJ_RADIX_3D_Y_4_2 = legate_numpy.NUMPY_PROJ_RADIX_3D_Y_4_2
    PROJ_RADIX_3D_Y_4_3 = legate_numpy.NUMPY_PROJ_RADIX_3D_Y_4_3
    PROJ_RADIX_3D_Z_4_0 = legate_numpy.NUMPY_PROJ_RADIX_3D_Z_4_0
    PROJ_RADIX_3D_Z_4_1 = legate_numpy.NUMPY_PROJ_RADIX_3D_Z_4_1
    PROJ_RADIX_3D_Z_4_2 = legate_numpy.NUMPY_PROJ_RADIX_3D_Z_4_2
    PROJ_RADIX_3D_Z_4_3 = legate_numpy.NUMPY_PROJ_RADIX_3D_Z_4_3
    # Flattening
    PROJ_ND_1D_C_ORDER = legate_numpy.NUMPY_PROJ_ND_1D_C_ORDER
    # Must always be last
    PROJ_LAST = legate_numpy.NUMPY_PROJ_LAST
