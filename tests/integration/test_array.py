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

SCALARS = (
    0,
    -10.5,
    1 + 1j,
)

ARRAYS = (
    [],
    (1, 2),
    ((1, 2),),
    [(1, 2), (3, 4.1)],
    (
        [1, 2.1],
        [3, 4 + 4j],
    ),
)


def strict_type_equal(a, b):
    return np.array_equal(a, b) and a.dtype == b.dtype


@pytest.mark.parametrize(
    "object",
    (None,) + SCALARS + ARRAYS,
    ids=lambda object: f"(object={object})",
)
def test_array_basic(object):
    res_np = np.array(object)
    res_num = num.array(object)
    assert strict_type_equal(res_np, res_num)


def test_array_ndarray():
    object = [[1, 2], [3, 4]]
    res_np = np.array(np.array(object))
    res_num = num.array(num.array(object))
    assert strict_type_equal(res_np, res_num)


DTYPES = (
    np.int32,
    np.float64,
    np.complex128,
)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: f"(dtype={dtype})")
@pytest.mark.parametrize(
    "object",
    (0, -10.5, [], [1, 2], [[1, 2], [3, 4.1]]),
    ids=lambda object: f"(object={object})",
)
def test_array_dtype(object, dtype):
    res_np = np.array(object, dtype=dtype)
    res_num = num.array(object, dtype=dtype)
    assert strict_type_equal(res_np, res_num)


@pytest.mark.parametrize(
    "ndmin",
    range(-1, LEGATE_MAX_DIM + 1),
    ids=lambda ndmin: f"(ndmin={ndmin})",
)
@pytest.mark.parametrize(
    "object",
    (0, [], [1, 2], [[1, 2], [3, 4.1]]),
    ids=lambda object: f"(object={object})",
)
def test_array_ndmin(object, ndmin):
    res_np = np.array(object, ndmin=ndmin)
    res_num = num.array(object, ndmin=ndmin)
    assert strict_type_equal(res_np, res_num)


@pytest.mark.parametrize(
    "copy", (True, False), ids=lambda copy: f"(copy={copy})"
)
def test_array_copy(copy):
    x = [[1, 2, 3], [4, 5, 6]]
    x_np = np.array(x)
    xc_np = np.array(x_np, copy=copy)
    x_np[0, :] = [7, 8, 9]

    x_num = num.array(x)
    xc_num = num.array(x_num, copy=copy)
    x_num[0, :] = [7, 8, 9]

    assert strict_type_equal(xc_np, xc_num)


class TestArrayErrors:
    @pytest.mark.parametrize(
        "dtype", (np.int32, np.float64), ids=lambda dtype: f"(dtype={dtype})"
    )
    @pytest.mark.parametrize(
        "object",
        (1 + 1j, [1, 2, 3.0, 4 + 4j]),
        ids=lambda object: f"(object={object})",
    )
    def test_invalid_dtype(self, object, dtype):
        expected_exc = TypeError
        with pytest.raises(expected_exc):
            np.array(object, dtype=dtype)
        with pytest.raises(expected_exc):
            num.array(object, dtype=dtype)


@pytest.mark.parametrize(
    "object",
    (None,) + SCALARS + ARRAYS,
    ids=lambda object: f"(object={object})",
)
def test_asarray_basic(object):
    res_np = np.asarray(object)
    res_num = num.asarray(object)
    assert strict_type_equal(res_np, res_num)


def test_asarray_ndarray():
    object = [[1, 2], [3, 4]]
    res_np = np.asarray(np.array(object))
    res_num = num.asarray(num.array(object))
    assert strict_type_equal(res_np, res_num)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: f"(dtype={dtype})")
@pytest.mark.parametrize(
    "object",
    (0, -10.5, [], [1, 2], [[1, 2], [3, 4.1]]),
    ids=lambda object: f"(object={object})",
)
def test_asarray_dtype(object, dtype):
    res_np = np.asarray(object, dtype=dtype)
    res_num = num.asarray(object, dtype=dtype)
    assert strict_type_equal(res_np, res_num)


class TestAsArrayErrors:
    @pytest.mark.parametrize(
        "dtype", (np.int32, np.float64), ids=lambda dtype: f"(dtype={dtype})"
    )
    @pytest.mark.parametrize(
        "object",
        (1 + 1j, [1, 2, 3.0, 4 + 4j]),
        ids=lambda object: f"(object={object})",
    )
    def test_invalid_dtype(self, object, dtype):
        expected_exc = TypeError
        with pytest.raises(expected_exc):
            np.asarray(object, dtype=dtype)
        with pytest.raises(expected_exc):
            num.asarray(object, dtype=dtype)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
