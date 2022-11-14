# Copyright 2022 NVIDIA Corporation
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

import numpy as np
import pytest
from legate.core import LEGATE_MAX_DIM

import cunumeric as num

np.random.seed(42)


def compare_assert(a_np, a_num):
    if not num.array_equal(a_np, a_num):
        print(f"fnumpy, shape {a_np.shape}:")
        print(a_np)
        print(f"cuNumeric, shape {a_num.shape}:")
        print(a_num)
        assert False


def generate_random(volume, datatype):
    a_np = None

    if np.issubdtype(datatype, np.integer):
        a_np = np.array(
            np.random.randint(
                np.iinfo(datatype).min,
                np.iinfo(datatype).max,
                size=volume,
                dtype=datatype,
            ),
            dtype=datatype,
        )
    elif np.issubdtype(datatype, np.floating):
        a_np = np.array(np.random.random(size=volume), dtype=datatype)
    elif np.issubdtype(datatype, np.complexfloating):
        a_np = np.array(
            np.random.random(size=volume) + np.random.random(size=volume) * 1j,
            dtype=datatype,
        )
    else:
        print(f"UNKNOWN type {datatype}")
        assert False
    return a_np


def check_api(a, dtype2=None, v=None):

    a_sorted = np.sort(a)
    a_argsorted = np.argsort(a)
    if v is None:
        if dtype2 is not None:
            v = generate_random(10, dtype2)
        else:
            v = generate_random(10, a.dtype)
    if v.size > 0 and v.ndim > 0:
        v_scalar = v[0]
    else:
        v_scalar = 0

    print("A=")
    print(a)
    print("A_sorted=")
    print(a_sorted)
    print("A_argsorted=")
    print(a_argsorted)
    print("V=")
    print(v)

    a_num = num.array(a)
    compare_assert(a, a_num)

    v_num = num.array(v)
    compare_assert(v, v_num)

    a_num_sorted = num.array(a_sorted)
    compare_assert(a_sorted, a_num_sorted)

    a_num_argsorted = num.array(a_argsorted)
    compare_assert(a_argsorted, a_num_argsorted)

    # left, array
    print("Left/Array 1")
    compare_assert(a_sorted.searchsorted(v), a_num_sorted.searchsorted(v_num))
    print("Left/Array 2")
    compare_assert(
        np.searchsorted(a_sorted, v), num.searchsorted(a_num_sorted, v_num)
    )

    # left, scalar
    print("Left/Scalar 1")
    compare_assert(
        a_sorted.searchsorted(v_scalar), a_num_sorted.searchsorted(v_scalar)
    )
    print("Left/Scalar 2")
    compare_assert(
        np.searchsorted(a_sorted, v_scalar),
        num.searchsorted(a_num_sorted, v_scalar),
    )

    # left, array, sorter
    print("Left/Array/Sorter 1")
    compare_assert(
        a.searchsorted(v, sorter=a_argsorted),
        a_num.searchsorted(v_num, sorter=a_num_argsorted),
    )
    print("Left/Array/Sorter 2")
    compare_assert(
        np.searchsorted(a, v, sorter=a_argsorted),
        num.searchsorted(a_num, v_num, sorter=a_num_argsorted),
    )

    # right, array
    print("Right/Array 1")
    compare_assert(
        a_sorted.searchsorted(v, side="right"),
        a_num_sorted.searchsorted(v_num, side="right"),
    )
    print("Right/Array 2")
    compare_assert(
        np.searchsorted(a_sorted, v, side="right"),
        num.searchsorted(a_num_sorted, v_num, side="right"),
    )

    # right, scalar
    print("Right/Scalar 1")
    compare_assert(
        a_sorted.searchsorted(v_scalar, side="right"),
        a_num_sorted.searchsorted(v_scalar, side="right"),
    )
    print("Right/Scalar 2")
    compare_assert(
        np.searchsorted(a_sorted, v_scalar, side="right"),
        num.searchsorted(a_num_sorted, v_scalar, side="right"),
    )

    # right, array, sorter
    print("Right/Array/Sorter 1")
    compare_assert(
        a.searchsorted(v, side="right", sorter=a_argsorted),
        a_num.searchsorted(v_num, side="right", sorter=a_num_argsorted),
    )
    print("Right/Array/Sorter 2")
    compare_assert(
        np.searchsorted(a, v, side="right", sorter=a_argsorted),
        num.searchsorted(a_num, v_num, side="right", sorter=a_num_argsorted),
    )


STANDARD_CASES = [
    (156, np.uint8),
    (123, np.uint16),
    (241, np.uint32),
    (1, np.uint64),
    (21, np.int8),
    (5, np.int16),
    (34, np.int32),
    (11, np.int64),
    (31, np.float32),
    (11, np.float64),
    (422, np.double),
    (220, np.double),
    (244, np.complex64),
    (24, np.complex128),
    (220, np.complex128),
    (0, np.uint32),
]

DTYPE_CASES = [
    (3, np.uint64, np.float32),
    (51, np.uint32, np.complex64),
    (23, np.uint32, np.float64),
    (51, np.complex64, np.float64),
    (21, np.complex64, np.int32),
    (22, np.complex128, np.float32),
]


def test_simple():
    check_api(np.arange(25))


def test_empty_v():
    check_api(np.arange(25), None, np.arange(0))
    check_api(np.array([]))
    check_api(np.arange(0), None, np.arange(0))


@pytest.mark.parametrize("volume, dtype1, dtype2", DTYPE_CASES, ids=str)
def test_dtype_conversions(volume, dtype1, dtype2):
    print(f"---- check for dtype {dtype1} (a) and {dtype2} (v)")
    check_api(generate_random(volume, dtype1), dtype2)


@pytest.mark.parametrize("volume, dtype", STANDARD_CASES, ids=str)
def test_standard_cases(volume, dtype):
    print(f"---- check for dtype {dtype}")
    check_api(generate_random(volume, dtype))


@pytest.mark.parametrize("ndim", range(0, LEGATE_MAX_DIM + 1))
def test_ndim(ndim):
    a = np.random.randint(-100, 100, size=100)
    v = np.random.randint(-100, 100, size=2**ndim).reshape(
        tuple(2 for i in range(ndim))
    )
    check_api(a, None, v)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
