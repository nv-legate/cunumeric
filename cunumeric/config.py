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
import numpy as np
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
    CHOOSE = _cunumeric.CUNUMERIC_CHOOSE
    CONTRACT = _cunumeric.CUNUMERIC_CONTRACT
    CONVERT = _cunumeric.CUNUMERIC_CONVERT
    CONVOLVE = _cunumeric.CUNUMERIC_CONVOLVE
    DIAG = _cunumeric.CUNUMERIC_DIAG
    DOT = _cunumeric.CUNUMERIC_DOT
    EYE = _cunumeric.CUNUMERIC_EYE
    FFT = _cunumeric.CUNUMERIC_FFT
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
    SORT = _cunumeric.CUNUMERIC_SORT
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


# Match these to CuNumericUnaryOpCode in cunumeric_c.h
@unique
class UnaryOpCode(IntEnum):
    ABSOLUTE = _cunumeric.CUNUMERIC_UOP_ABSOLUTE
    ARCCOS = _cunumeric.CUNUMERIC_UOP_ARCCOS
    ARCCOSH = _cunumeric.CUNUMERIC_UOP_ARCCOSH
    ARCSIN = _cunumeric.CUNUMERIC_UOP_ARCSIN
    ARCSINH = _cunumeric.CUNUMERIC_UOP_ARCSINH
    ARCTAN = _cunumeric.CUNUMERIC_UOP_ARCTAN
    ARCTANH = _cunumeric.CUNUMERIC_UOP_ARCTANH
    CBRT = _cunumeric.CUNUMERIC_UOP_CBRT
    CEIL = _cunumeric.CUNUMERIC_UOP_CEIL
    CLIP = _cunumeric.CUNUMERIC_UOP_CLIP
    CONJ = _cunumeric.CUNUMERIC_UOP_CONJ
    COPY = _cunumeric.CUNUMERIC_UOP_COPY
    COS = _cunumeric.CUNUMERIC_UOP_COS
    COSH = _cunumeric.CUNUMERIC_UOP_COSH
    DEG2RAD = _cunumeric.CUNUMERIC_UOP_DEG2RAD
    EXP = _cunumeric.CUNUMERIC_UOP_EXP
    EXP2 = _cunumeric.CUNUMERIC_UOP_EXP2
    EXPM1 = _cunumeric.CUNUMERIC_UOP_EXPM1
    FLOOR = _cunumeric.CUNUMERIC_UOP_FLOOR
    GETARG = _cunumeric.CUNUMERIC_UOP_GETARG
    IMAG = _cunumeric.CUNUMERIC_UOP_IMAG
    INVERT = _cunumeric.CUNUMERIC_UOP_INVERT
    ISFINITE = _cunumeric.CUNUMERIC_UOP_ISFINITE
    ISINF = _cunumeric.CUNUMERIC_UOP_ISINF
    ISNAN = _cunumeric.CUNUMERIC_UOP_ISNAN
    LOG = _cunumeric.CUNUMERIC_UOP_LOG
    LOG10 = _cunumeric.CUNUMERIC_UOP_LOG10
    LOG1P = _cunumeric.CUNUMERIC_UOP_LOG1P
    LOG2 = _cunumeric.CUNUMERIC_UOP_LOG2
    LOGICAL_NOT = _cunumeric.CUNUMERIC_UOP_LOGICAL_NOT
    NEGATIVE = _cunumeric.CUNUMERIC_UOP_NEGATIVE
    POSITIVE = _cunumeric.CUNUMERIC_UOP_POSITIVE
    RAD2DEG = _cunumeric.CUNUMERIC_UOP_RAD2DEG
    REAL = _cunumeric.CUNUMERIC_UOP_REAL
    RECIPROCAL = _cunumeric.CUNUMERIC_UOP_RECIPROCAL
    RINT = _cunumeric.CUNUMERIC_UOP_RINT
    SIGN = _cunumeric.CUNUMERIC_UOP_SIGN
    SIGNBIT = _cunumeric.CUNUMERIC_UOP_SIGNBIT
    SIN = _cunumeric.CUNUMERIC_UOP_SIN
    SINH = _cunumeric.CUNUMERIC_UOP_SINH
    SQUARE = _cunumeric.CUNUMERIC_UOP_SQUARE
    SQRT = _cunumeric.CUNUMERIC_UOP_SQRT
    TAN = _cunumeric.CUNUMERIC_UOP_TAN
    TANH = _cunumeric.CUNUMERIC_UOP_TANH
    TRUNC = _cunumeric.CUNUMERIC_UOP_TRUNC


# Match these to CuNumericRedopCode in cunumeric_c.h
@unique
class UnaryRedCode(IntEnum):
    ALL = _cunumeric.CUNUMERIC_RED_ALL
    ANY = _cunumeric.CUNUMERIC_RED_ANY
    ARGMAX = _cunumeric.CUNUMERIC_RED_ARGMAX
    ARGMIN = _cunumeric.CUNUMERIC_RED_ARGMIN
    CONTAINS = _cunumeric.CUNUMERIC_RED_CONTAINS
    COUNT_NONZERO = _cunumeric.CUNUMERIC_RED_COUNT_NONZERO
    MAX = _cunumeric.CUNUMERIC_RED_MAX
    MIN = _cunumeric.CUNUMERIC_RED_MIN
    PROD = _cunumeric.CUNUMERIC_RED_PROD
    SUM = _cunumeric.CUNUMERIC_RED_SUM


# Match these to CuNumericBinaryOpCode in cunumeric_c.h
@unique
class BinaryOpCode(IntEnum):
    ADD = _cunumeric.CUNUMERIC_BINOP_ADD
    ALLCLOSE = _cunumeric.CUNUMERIC_BINOP_ALLCLOSE
    ARCTAN2 = _cunumeric.CUNUMERIC_BINOP_ARCTAN2
    BITWISE_AND = _cunumeric.CUNUMERIC_BINOP_BITWISE_AND
    BITWISE_OR = _cunumeric.CUNUMERIC_BINOP_BITWISE_OR
    BITWISE_XOR = _cunumeric.CUNUMERIC_BINOP_BITWISE_XOR
    COPYSIGN = _cunumeric.CUNUMERIC_BINOP_COPYSIGN
    DIVIDE = _cunumeric.CUNUMERIC_BINOP_DIVIDE
    EQUAL = _cunumeric.CUNUMERIC_BINOP_EQUAL
    FLOAT_POWER = _cunumeric.CUNUMERIC_BINOP_FLOAT_POWER
    FLOOR_DIVIDE = _cunumeric.CUNUMERIC_BINOP_FLOOR_DIVIDE
    FMOD = _cunumeric.CUNUMERIC_BINOP_FMOD
    GCD = _cunumeric.CUNUMERIC_BINOP_GCD
    GREATER = _cunumeric.CUNUMERIC_BINOP_GREATER
    GREATER_EQUAL = _cunumeric.CUNUMERIC_BINOP_GREATER_EQUAL
    HYPOT = _cunumeric.CUNUMERIC_BINOP_HYPOT
    LCM = _cunumeric.CUNUMERIC_BINOP_LCM
    LEFT_SHIFT = _cunumeric.CUNUMERIC_BINOP_LEFT_SHIFT
    LESS = _cunumeric.CUNUMERIC_BINOP_LESS
    LESS_EQUAL = _cunumeric.CUNUMERIC_BINOP_LESS_EQUAL
    LOGADDEXP = _cunumeric.CUNUMERIC_BINOP_LOGADDEXP
    LOGADDEXP2 = _cunumeric.CUNUMERIC_BINOP_LOGADDEXP2
    LOGICAL_AND = _cunumeric.CUNUMERIC_BINOP_LOGICAL_AND
    LOGICAL_OR = _cunumeric.CUNUMERIC_BINOP_LOGICAL_OR
    LOGICAL_XOR = _cunumeric.CUNUMERIC_BINOP_LOGICAL_XOR
    MAXIMUM = _cunumeric.CUNUMERIC_BINOP_MAXIMUM
    MINIMUM = _cunumeric.CUNUMERIC_BINOP_MINIMUM
    MOD = _cunumeric.CUNUMERIC_BINOP_MOD
    MULTIPLY = _cunumeric.CUNUMERIC_BINOP_MULTIPLY
    NEXTAFTER = _cunumeric.CUNUMERIC_BINOP_NEXTAFTER
    NOT_EQUAL = _cunumeric.CUNUMERIC_BINOP_NOT_EQUAL
    POWER = _cunumeric.CUNUMERIC_BINOP_POWER
    RIGHT_SHIFT = _cunumeric.CUNUMERIC_BINOP_RIGHT_SHIFT
    SUBTRACT = _cunumeric.CUNUMERIC_BINOP_SUBTRACT


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


# Match these to fftType in fft_util.h
@unique
class FFTCode(IntEnum):
    FFT_R2C = 0x2a
    FFT_C2R = 0x2c
    FFT_C2C = 0x29
    FFT_D2Z = 0x6a
    FFT_Z2D = 0x6c
    FFT_Z2Z = 0x69

    @staticmethod
    def real_to_complex_code(dtype):
        if dtype == np.float64:
            return FFTCode.FFT_D2Z
        elif dtype == np.float32:
            return FFTCode.FFT_R2C
        else:
            raise TypeError("Data type for FFT not supported (supported types are float32 and float64)")

    @staticmethod
    def complex_to_real_code(dtype):
        if dtype == np.complex128:
            return FFTCode.FFT_Z2D
        elif dtype == np.complex64:
            return FFTCode.FFT_C2R
        else:
            raise TypeError("Data type for FFT not supported (supported types are complex64 and complex128)")

    @property
    def complex(self):
        if self == FFTCode.FFT_D2Z or self == FFTCode.FFT_Z2D:
            return FFTCode.FFT_Z2Z
        elif self == FFTCode.FFT_R2C or self == FFTCode.FFT_C2R:
            return FFTCode.FFT_C2C
        return self

    @property
    def input_dtype(self):
        if self == FFTCode.FFT_D2Z:
            return np.float64
        elif self == FFTCode.FFT_R2C:
            return np.float32
        elif self == FFTCode.FFT_Z2D:
            return np.complex128
        elif self == FFTCode.FFT_C2R:
            return np.complex64
        elif self == FFTCode.FFT_Z2Z:
            return np.complex128
        elif self == FFTCode.FFT_C2C:
            return np.complex64
        else:
            raise TypeError("Type of FFT not supported (supported types are C2C, R2C, C2R, Z2Z, D2Z, Z2D)")

    @property
    def output_dtype(self):
        if self == FFTCode.FFT_D2Z:
            return np.complex128
        elif self == FFTCode.FFT_R2C:
            return np.complex64
        elif self == FFTCode.FFT_Z2D:
            return np.float64
        elif self == FFTCode.FFT_C2R:
            return np.float32
        elif self == FFTCode.FFT_Z2Z:
            return np.complex128
        elif self == FFTCode.FFT_C2C:
            return np.complex64
        else:
            raise TypeError("Type of FFT not supported (supported types are C2C, R2C, C2R, Z2Z, D2Z, Z2D)")

@unique
class FFTDirection(IntEnum):
    FORWARD = -1
    INVERSE =  1

@unique
class FFTNormalization(IntEnum):
    FORWARD    =  1
    INVERSE    =  2
    ORTHOGONAL =  3
    
    @staticmethod
    def from_string(in_string):
        if in_string == 'forward':
            return FFTNormalization.FORWARD
        elif in_string == 'ortho':
            return FFTNormalization.ORTHOGONAL
        elif in_string == 'backward' or in_string is None:
            return FFTNormalization.INVERSE
        else:
            return None

    @staticmethod
    def reverse(in_string):
        if in_string == 'forward':
            return 'backward'
        elif in_string == 'backward' or in_string is None:
            return 'forward'
        else:
            return in_string
