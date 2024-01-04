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

import pytest
from legate.core import Library
from legate.core.context import Context
from mock import patch

import cunumeric.config as m  # module under test
from cunumeric import runtime


class _FakeSO:
    CUNUMERIC_MAX_TASKS = 10
    CUNUMERIC_MAX_MAPPERS = 20
    CUNUMERIC_MAX_REDOPS = 30


class TestCuNumericLib:
    def test___init__(self) -> None:
        lib = m.CuNumericLib("foo")
        assert isinstance(lib, Library)
        assert lib.name == "foo"
        assert lib.shared_object is None
        assert lib.runtime is None

    def test_get_name(self) -> None:
        lib = m.CuNumericLib("foo")
        assert lib.get_name() == "foo"

    def test_get_shared_library(self) -> None:
        lib = m.CuNumericLib("foo")
        result = lib.get_shared_library()
        assert isinstance(result, str)

        from cunumeric.install_info import libpath

        assert result.startswith(libpath)

        assert "libcunumeric" in result

        assert result.endswith(lib.get_library_extension())

    def test_get_c_header(self) -> None:
        lib = m.CuNumericLib("foo")

        from cunumeric.install_info import header

        assert lib.get_c_header() == header

    def test_get_registration_callback(self) -> None:
        lib = m.CuNumericLib("foo")
        assert (
            lib.get_registration_callback() == "cunumeric_perform_registration"
        )

    def test_initialize(self) -> None:
        lib = m.CuNumericLib("foo")
        lib.initialize(_FakeSO)
        assert lib.shared_object == _FakeSO

        # error if runtime already set
        lib.runtime = runtime
        with pytest.raises(AssertionError):
            lib.initialize(_FakeSO)

    def test_set_runtine(self) -> None:
        lib = m.CuNumericLib("foo")

        # error if not initialized
        with pytest.raises(AssertionError):
            lib.set_runtime(runtime)

        lib.initialize(_FakeSO)
        lib.set_runtime(runtime)
        assert lib.runtime == runtime

        # error if runtime already set
        with pytest.raises(AssertionError):
            lib.set_runtime(runtime)

    @patch("cunumeric.runtime.destroy")
    def test_destroy(self, mock_destroy) -> None:
        lib = m.CuNumericLib("foo")
        lib.initialize(_FakeSO)
        lib.set_runtime(runtime)
        lib.destroy()
        mock_destroy.assert_called_once_with()


def test_CUNUMERIC_LIB_NAME() -> None:
    assert m.CUNUMERIC_LIB_NAME == "cunumeric"


def test_cunumeric_lib() -> None:
    assert isinstance(m.cunumeric_lib, m.CuNumericLib)


def test_cunumeric_context() -> None:
    assert isinstance(m.cunumeric_context, Context)


def test_CuNumericOpCode() -> None:
    assert set(m.CuNumericOpCode.__members__) == {
        "ADVANCED_INDEXING",
        "ARANGE",
        "ARGWHERE",
        "BATCHED_CHOLESKY",
        "BINARY_OP",
        "BINARY_RED",
        "BINCOUNT",
        "BITGENERATOR",
        "CHOOSE",
        "CONTRACT",
        "CONVERT",
        "CONVOLVE",
        "DIAG",
        "DOT",
        "EYE",
        "FFT",
        "FILL",
        "FLIP",
        "GEMM",
        "HISTOGRAM",
        "LOAD_CUDALIBS",
        "MATMUL",
        "MATVECMUL",
        "NONZERO",
        "PACKBITS",
        "POTRF",
        "PUTMASK",
        "RAND",
        "READ",
        "REPEAT",
        "SELECT",
        "SCALAR_UNARY_RED",
        "SCAN_GLOBAL",
        "SCAN_LOCAL",
        "SOLVE",
        "SORT",
        "SEARCHSORTED",
        "SYRK",
        "TILE",
        "TRANSPOSE_COPY_2D",
        "TRILU",
        "TRSM",
        "UNARY_OP",
        "UNARY_RED",
        "UNIQUE",
        "UNIQUE_REDUCE",
        "UNLOAD_CUDALIBS",
        "UNPACKBITS",
        "WHERE",
        "WINDOW",
        "WRAP",
        "WRITE",
        "ZIP",
    }


def test_UnaryOpCode() -> None:
    assert (set(m.UnaryOpCode.__members__)) == {
        "ABSOLUTE",
        "ARCCOS",
        "ARCCOSH",
        "ARCSIN",
        "ARCSINH",
        "ARCTAN",
        "ARCTANH",
        "CBRT",
        "CEIL",
        "CLIP",
        "CONJ",
        "COPY",
        "COS",
        "COSH",
        "DEG2RAD",
        "EXP",
        "EXP2",
        "EXPM1",
        "FLOOR",
        "FREXP",
        "GETARG",
        "IMAG",
        "INVERT",
        "ISFINITE",
        "ISINF",
        "ISNAN",
        "LOG",
        "LOG10",
        "LOG1P",
        "LOG2",
        "LOGICAL_NOT",
        "MODF",
        "NEGATIVE",
        "POSITIVE",
        "RAD2DEG",
        "REAL",
        "RECIPROCAL",
        "RINT",
        "SIGN",
        "SIGNBIT",
        "SIN",
        "SINH",
        "SQRT",
        "SQUARE",
        "TAN",
        "TANH",
        "TRUNC",
    }


def test_RandGenCode() -> None:
    assert (set(m.RandGenCode.__members__)) == {"UNIFORM", "NORMAL", "INTEGER"}


def test_CuNumericTunable() -> None:
    assert (set(m.CuNumericTunable.__members__)) == {
        "NUM_GPUS",
        "NUM_PROCS",
        "MAX_EAGER_VOLUME",
    }


def test_ScanCode() -> None:
    assert (set(m.ScanCode.__members__)) == {"PROD", "SUM"}


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
