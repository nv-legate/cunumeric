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
    ARANGE = _cunumeric.CUNUMERIC_ARANGE
    BINARY_OP = _cunumeric.CUNUMERIC_BINARY_OP
    BINARY_RED = _cunumeric.CUNUMERIC_BINARY_RED
    BINCOUNT = _cunumeric.CUNUMERIC_BINCOUNT
    CONVERT = _cunumeric.CUNUMERIC_CONVERT
    DIAG = _cunumeric.CUNUMERIC_DIAG
    DOT = _cunumeric.CUNUMERIC_DOT
    EYE = _cunumeric.CUNUMERIC_EYE
    FILL = _cunumeric.CUNUMERIC_FILL
    MATMUL = _cunumeric.CUNUMERIC_MATMUL
    MATVECMUL = _cunumeric.CUNUMERIC_MATVECMUL
    NONZERO = _cunumeric.CUNUMERIC_NONZERO
    RAND = _cunumeric.CUNUMERIC_RAND
    READ = _cunumeric.CUNUMERIC_READ
    SCALAR_UNARY_RED = _cunumeric.CUNUMERIC_SCALAR_UNARY_RED
    TILE = _cunumeric.CUNUMERIC_TILE
    TRANSPOSE = _cunumeric.CUNUMERIC_TRANSPOSE
    UNARY_OP = _cunumeric.CUNUMERIC_UNARY_OP
    UNARY_RED = _cunumeric.CUNUMERIC_UNARY_RED
    WHERE = _cunumeric.CUNUMERIC_WHERE
    WRITE = _cunumeric.CUNUMERIC_WRITE


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


# Match these to CuNumericRedopID in cunumeric_c.h
@unique
class CuNumericRedopCode(IntEnum):
    ARGMAX = 1
    ARGMIN = 2


# Match these to CuNumericTunable in cunumeric_c.h
@unique
class CuNumericTunable(IntEnum):
    NUM_GPUS = _cunumeric.CUNUMERIC_TUNABLE_NUM_GPUS
    MAX_EAGER_VOLUME = _cunumeric.CUNUMERIC_TUNABLE_MAX_EAGER_VOLUME
