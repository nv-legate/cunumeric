# Copyright 2021-2022 NVIDIA Corporation
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

import os
from enum import IntEnum, unique

from legate.core import Library, ResourceConfig, get_legate_runtime


# Load the cuNumeric library first so we have a shard object that
# we can use to initialize all these configuration enumerations
class CuNumericLib(Library):
    def __init__(self, name):
        self.name = name
        self.runtime = None
        self.shared_object = None

    def get_name(self):
        return self.name

    def get_shared_library(self):
        from cunumeric.install_info import libpath

        return os.path.join(
            libpath, "libcunumeric" + self.get_library_extension()
        )

    def get_c_header(self):
        from cunumeric.install_info import header

        return header

    def get_registration_callback(self):
        return "cunumeric_perform_registration"

    def initialize(self, shared_object):
        assert self.runtime is None
        self.shared_object = shared_object

    def set_runtime(self, runtime):
        assert self.runtime is None
        assert self.shared_object is not None
        self.runtime = runtime

    def get_resource_configuration(self):
        assert self.shared_object is not None
        config = ResourceConfig()
        config.max_tasks = self.shared_object.CUNUMERIC_MAX_TASKS
        config.max_mappers = self.shared_object.CUNUMERIC_MAX_MAPPERS
        config.max_reduction_ops = self.shared_object.CUNUMERIC_MAX_REDOPS
        config.max_projections = 0
        config.max_shardings = 0
        return config

    def destroy(self):
        if self.runtime is not None:
            self.runtime.destroy()


CUNUMERIC_LIB_NAME = "cunumeric"
cunumeric_lib = CuNumericLib(CUNUMERIC_LIB_NAME)
cunumeric_context = get_legate_runtime().register_library(cunumeric_lib)
_cunumeric = cunumeric_lib.shared_object


# Match these to CuNumericOpCode in cunumeric_c.h
@unique
class CuNumericOpCode(IntEnum):
    ADVANCED_INDX = _cunumeric.CUNUMERIC_ADVANCED_INDEXING
    ARANGE = _cunumeric.CUNUMERIC_ARANGE
    BINARY_OP = _cunumeric.CUNUMERIC_BINARY_OP
    BINARY_RED = _cunumeric.CUNUMERIC_BINARY_RED
    BINCOUNT = _cunumeric.CUNUMERIC_BINCOUNT
    CHOOSE = _cunumeric.CUNUMERIC_CHOOSE
    CONTRACT = _cunumeric.CUNUMERIC_CONTRACT
    CONVERT = _cunumeric.CUNUMERIC_CONVERT
    CONVOLVE = _cunumeric.CUNUMERIC_CONVOLVE
    DIAG = _cunumeric.CUNUMERIC_DIAG
    DOT = _cunumeric.CUNUMERIC_DOT
    EYE = _cunumeric.CUNUMERIC_EYE
    FILL = _cunumeric.CUNUMERIC_FILL
    FLIP = _cunumeric.CUNUMERIC_FLIP
    GEMM = _cunumeric.CUNUMERIC_GEMM
    LOAD_CUDALIBS = _cunumeric.CUNUMERIC_LOAD_CUDALIBS
    MATMUL = _cunumeric.CUNUMERIC_MATMUL
    MATVECMUL = _cunumeric.CUNUMERIC_MATVECMUL
    NONZERO = _cunumeric.CUNUMERIC_NONZERO
    POTRF = _cunumeric.CUNUMERIC_POTRF
    RAND = _cunumeric.CUNUMERIC_RAND
    READ = _cunumeric.CUNUMERIC_READ
    REPEAT = _cunumeric.CUNUMERIC_REPEAT
    SCALAR_UNARY_RED = _cunumeric.CUNUMERIC_SCALAR_UNARY_RED
    SYRK = _cunumeric.CUNUMERIC_SYRK
    TILE = _cunumeric.CUNUMERIC_TILE
    TRANSPOSE_COPY_2D = _cunumeric.CUNUMERIC_TRANSPOSE_COPY_2D
    TRILU = _cunumeric.CUNUMERIC_TRILU
    TRSM = _cunumeric.CUNUMERIC_TRSM
    UNARY_OP = _cunumeric.CUNUMERIC_UNARY_OP
    UNARY_RED = _cunumeric.CUNUMERIC_UNARY_RED
    UNIQUE = _cunumeric.CUNUMERIC_UNIQUE
    UNIQUE_REDUCE = _cunumeric.CUNUMERIC_UNIQUE_REDUCE
    UNLOAD_CUDALIBS = _cunumeric.CUNUMERIC_UNLOAD_CUDALIBS
    WHERE = _cunumeric.CUNUMERIC_WHERE
    WRITE = _cunumeric.CUNUMERIC_WRITE
    ZIP = _cunumeric.CUNUMERIC_ZIP


# Match these to BinaryOpCode in binary_op_util.h
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
    LOGICAL_AND = 9
    LOGICAL_OR = 10
    LOGICAL_XOR = 11
    MAXIMUM = 12
    MINIMUM = 13
    MOD = 14
    MULTIPLY = 15
    NOT_EQUAL = 16
    POWER = 17
    SUBTRACT = 18
    ALLCLOSE = 19


# Match these to UnaryOpCode in unary_op_util.h
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
    EXP2 = 10
    FLOOR = 11
    INVERT = 12
    ISINF = 13
    ISNAN = 14
    LOG = 15
    LOG10 = 16
    LOGICAL_NOT = 17
    NEGATIVE = 18
    RINT = 19
    SIGN = 20
    SIN = 21
    SQRT = 22
    TAN = 23
    TANH = 24
    CONJ = 25
    REAL = 26
    IMAG = 27
    GETARG = 28


# Match these to UnaryRedCode in unary_red_util.h
@unique
class UnaryRedCode(IntEnum):
    ALL = 1
    ANY = 2
    MAX = 3
    MIN = 4
    PROD = 5
    SUM = 6
    ARGMAX = 7
    ARGMIN = 8
    CONTAINS = 9
    COUNT_NONZERO = 10


# Match these to RandGenCode in rand_util.h
@unique
class RandGenCode(IntEnum):
    UNIFORM = 1
    NORMAL = 2
    INTEGER = 3


# Match these to CuNumericRedopID in cunumeric_c.h
@unique
class CuNumericRedopCode(IntEnum):
    ARGMAX = 1
    ARGMIN = 2


# Match these to CuNumericTunable in cunumeric_c.h
@unique
class CuNumericTunable(IntEnum):
    NUM_GPUS = _cunumeric.CUNUMERIC_TUNABLE_NUM_GPUS
    NUM_PROCS = _cunumeric.CUNUMERIC_TUNABLE_NUM_PROCS
    MAX_EAGER_VOLUME = _cunumeric.CUNUMERIC_TUNABLE_MAX_EAGER_VOLUME
    HAS_NUMAMEM = _cunumeric.CUNUMERIC_TUNABLE_HAS_NUMAMEM
