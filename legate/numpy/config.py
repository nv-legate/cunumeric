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
from enum import IntEnum, unique

import numpy as np

from legate.core import LegateLibrary, legate_add_library, legion


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


# Match these to NumPyOpCode in legate_numpy_c.h
@unique
class NumPyOpCode(IntEnum):
    TILE = legate_numpy.NUMPY_TILE
    # Type-erased operators
    BINARY_OP = legate_numpy.NUMPY_BINARY_OP
    SCALAR_BINARY_OP = legate_numpy.NUMPY_SCALAR_BINARY_OP
    FILL = legate_numpy.NUMPY_FILL
    SCALAR_UNARY_RED = legate_numpy.NUMPY_SCALAR_UNARY_RED
    UNARY_RED = legate_numpy.NUMPY_UNARY_RED
    UNARY_OP = legate_numpy.NUMPY_UNARY_OP
    SCALAR_UNARY_OP = legate_numpy.NUMPY_SCALAR_UNARY_OP
    BINARY_RED = legate_numpy.NUMPY_BINARY_RED
    CONVERT = legate_numpy.NUMPY_CONVERT
    SCALAR_CONVERT = legate_numpy.NUMPY_SCALAR_CONVERT
    WHERE = legate_numpy.NUMPY_WHERE
    SCALAR_WHERE = legate_numpy.NUMPY_SCALAR_WHERE
    READ = legate_numpy.NUMPY_READ
    WRITE = legate_numpy.NUMPY_WRITE
    DIAG = legate_numpy.NUMPY_DIAG
    MATMUL = legate_numpy.NUMPY_MATMUL
    MATVECMUL = legate_numpy.NUMPY_MATVECMUL
    DOT = legate_numpy.NUMPY_DOT
    BINCOUNT = legate_numpy.NUMPY_BINCOUNT
    EYE = legate_numpy.NUMPY_EYE
    RAND = legate_numpy.NUMPY_RAND
    ARANGE = legate_numpy.NUMPY_ARANGE
    TRANSPOSE = legate_numpy.NUMPY_TRANSPOSE


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
    ALLCLOSE = 16


@unique
class UnaryOpCode(IntEnum):
    ABSOLUTE = 1
    ARCCOS = 2
    ARCSIN = 3
    ARCTAN = 4
    CEIL = 5
    CLIP = 6
    COPY = 7
    COS = 8
    EXP = 9
    FLOOR = 10
    INVERT = 11
    ISINF = 12
    ISNAN = 13
    LOG = 14
    LOGICAL_NOT = 15
    NEGATIVE = 16
    SIN = 17
    SQRT = 18
    TAN = 19
    TANH = 20
    CONJ = 21
    REAL = 22
    IMAG = 23
    GETARG = 24


@unique
class UnaryRedCode(IntEnum):
    MAX = 1
    MIN = 2
    PROD = 3
    SUM = 4
    ARGMAX = 5
    ARGMIN = 6
    CONTAINS = 7
    COUNT_NONZERO = 8


@unique
class RandGenCode(IntEnum):
    UNIFORM = 1
    NORMAL = 2
    INTEGER = 3


# Match these to NumPyRedopID in legate_numpy_c.h
@unique
class NumPyRedopCode(IntEnum):
    ARGMIN_REDOP = legate_numpy.NUMPY_ARGMIN_REDOP
    ARGMAX_REDOP = legate_numpy.NUMPY_ARGMAX_REDOP


numpy_unary_reduction_op_offsets = {
    UnaryRedCode.SUM: legion.LEGION_REDOP_KIND_SUM,
    UnaryRedCode.PROD: legion.LEGION_REDOP_KIND_PROD,
    UnaryRedCode.MIN: legion.LEGION_REDOP_KIND_MIN,
    UnaryRedCode.MAX: legion.LEGION_REDOP_KIND_MAX,
    UnaryRedCode.ARGMAX: NumPyRedopCode.ARGMAX_REDOP,
    UnaryRedCode.ARGMIN: NumPyRedopCode.ARGMIN_REDOP,
}


def max_identity(ty):
    if ty.kind == "i" or ty.kind == "u":
        return np.iinfo(ty).min
    elif ty.kind == "f":
        return np.finfo(ty).min
    elif ty.kind == "c":
        return max_identity(np.float64) + max_identity(np.float64) * 1j
    elif ty.kind == "b":
        return False
    else:
        raise ValueError(f"Unsupported dtype: {ty}")


def min_identity(ty):
    if ty.kind == "i" or ty.kind == "u":
        return np.iinfo(ty).max
    elif ty.kind == "f":
        return np.finfo(ty).max
    elif ty.kind == "c":
        return min_identity(np.float64) + min_identity(np.float64) * 1j
    elif ty.kind == "b":
        return True
    else:
        raise ValueError(f"Unsupported dtype: {ty}")


numpy_unary_reduction_identities = {
    UnaryRedCode.SUM: lambda _: 0,
    UnaryRedCode.PROD: lambda _: 1,
    UnaryRedCode.MIN: min_identity,
    UnaryRedCode.MAX: max_identity,
    UnaryRedCode.ARGMAX: lambda ty: (np.iinfo(np.int64).min, max_identity(ty)),
    UnaryRedCode.ARGMIN: lambda ty: (np.iinfo(np.int64).min, min_identity(ty)),
}

numpy_scalar_reduction_op_offsets = {
    UnaryRedCode.MAX: legate_numpy.NUMPY_SCALAR_MAX_REDOP,
    UnaryRedCode.MIN: legate_numpy.NUMPY_SCALAR_MIN_REDOP,
    UnaryRedCode.PROD: legate_numpy.NUMPY_SCALAR_PROD_REDOP,
    UnaryRedCode.SUM: legate_numpy.NUMPY_SCALAR_SUM_REDOP,
    UnaryRedCode.CONTAINS: legate_numpy.NUMPY_SCALAR_SUM_REDOP,
    UnaryRedCode.ARGMAX: legate_numpy.NUMPY_SCALAR_ARGMAX_REDOP,
    UnaryRedCode.ARGMIN: legate_numpy.NUMPY_SCALAR_ARGMIN_REDOP,
    UnaryRedCode.COUNT_NONZERO: legate_numpy.NUMPY_SCALAR_SUM_REDOP,
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
    # Must always be last
    PROJ_LAST = legate_numpy.NUMPY_PROJ_LAST
