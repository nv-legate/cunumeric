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

from itertools import product

import numpy as np
import pytest

import cunumeric as num


def test_array():
    x = num.array([1, 2, 3])
    y = np.array([1, 2, 3])
    z = num.array(y)
    assert np.array_equal(x, z)
    assert x.dtype == z.dtype

    x = num.array([1, 2, 3])
    y = num.array(x)
    assert num.array_equal(x, y)
    assert x.dtype == y.dtype


CREATION_FUNCTIONS = ("empty", "zeros", "ones")
FILLED_VALUES = [0, 1, 1000, 123.456]
SIZES = (0, 1, 2)
NDIMS = 5
DTYPES = (np.uint32, np.int32, np.float64, np.complex128)


@pytest.mark.parametrize("fn", CREATION_FUNCTIONS)
def test_creation_func(fn):
    num_f = getattr(num, fn)
    np_f = getattr(np, fn)

    par = (SIZES, range(NDIMS), DTYPES)
    for size, ndims, dtype in product(*par):
        shape = ndims * [size]

        xf = num_f(shape, dtype=dtype)
        yf = np_f(shape, dtype=dtype)

        if fn == "empty":
            assert xf.shape == yf.shape
        else:
            assert np.array_equal(xf, yf)
        assert xf.dtype == yf.dtype


@pytest.mark.parametrize("value", FILLED_VALUES)
def test_full(value):
    par = (SIZES, range(NDIMS), DTYPES)
    for size, ndims, dtype in product(*par):
        shape = ndims * [size]

        xf = num.full(shape, value, dtype=dtype)
        yf = np.full(shape, value, dtype=dtype)

        assert np.array_equal(xf, yf)
        assert xf.dtype == yf.dtype


SHAPES_NEGATIVE = [
    -1,
    (-1, 2, 3),
    # num.array([2, -3, 4]),  ## it would raise RuntimeError("Unable to find attachment to remove") when num.array
    # is removed at the end as global variable
    np.array([2, -3, 4]),
]


def test_creation_func_negative():
    for shape in SHAPES_NEGATIVE + [num.array([2, -3, 4])]:
        with pytest.raises(ValueError):
            num.empty(shape)
        with pytest.raises(ValueError):
            num.zeros(shape)
        with pytest.raises(ValueError):
            num.ones(shape)
        with pytest.raises(ValueError):
            num.full(shape, 10)

    num.full((2, 3), [1])
    with pytest.raises(AssertionError):
        num.full((2, 3), [10, 20, 30])


DATA_ARGS = [
    # Array scalars
    (np.array(3.0), None),
    (np.array(3), "f8"),
    # 1D arrays
    (np.array([]), None),
    (np.arange(6, dtype="f4"), None),
    (np.arange(6), "c16"),
    # 2D arrays
    (np.array([[]]), None),
    (np.arange(6).reshape(2, 3), None),
    (np.arange(6).reshape(3, 2), "i1"),
    # 3D arrays
    (np.arange(24).reshape(2, 3, 4), None),
    (np.arange(24).reshape(4, 3, 2), "f4"),
]
LIKE_FUNCTIONS = ("empty_like", "zeros_like", "ones_like")


@pytest.mark.parametrize("x_np,dtype", DATA_ARGS)
@pytest.mark.parametrize("fn", LIKE_FUNCTIONS)
def test_func_like(fn, x_np, dtype):
    num_f = getattr(num, fn)
    np_f = getattr(np, fn)

    x = num.array(x_np)
    xfl = num_f(x, dtype=dtype)
    yfl = np_f(x_np, dtype=dtype)

    if fn == "empty_like":
        assert xfl.shape == yfl.shape
    else:
        assert np.array_equal(xfl, yfl)
    assert xfl.dtype == yfl.dtype


@pytest.mark.parametrize("value", FILLED_VALUES)
@pytest.mark.parametrize("x_np, dtype", DATA_ARGS)
def test_full_like(x_np, dtype, value):
    x = num.array(x_np)

    xfl = num.full_like(x, value, dtype=dtype)
    yfl = np.full_like(x_np, value, dtype=dtype)
    assert np.array_equal(xfl, yfl)
    assert xfl.dtype == yfl.dtype


def test_full_like_negative():
    x = num.array([[1, 2, 3], [4, 5, 6]])
    num.full_like(x, [1])
    with pytest.raises(AssertionError):
        num.full_like(x, [10, 20, 30])


ARANGE_ARGS = [
    (0,),
    (10,),
    (3.5,),
    (2, 10),
    (-2.5, 10.0),
    # (1, -10, -2.5), ### output: num: array([ 1, -1, -3, -5, -7]), np: array([ 1. , -1.5, -4. , -6.5, -9. ]
    (1.0, -10.0, -2.5),
    (-10, 10, 10),
]


@pytest.mark.parametrize("args", ARANGE_ARGS, ids=str)
def test_arange(args):
    x = num.arange(*args)
    y = np.arange(*args)
    assert np.array_equal(x, y)
    assert x.dtype == y.dtype


@pytest.mark.parametrize("dtype", [np.int32, np.float64], ids=str)
@pytest.mark.parametrize("args", ARANGE_ARGS, ids=str)
def test_arange_with_dtype(args, dtype):
    x = num.arange(*args, dtype=dtype)
    y = np.arange(*args, dtype=dtype)
    assert np.array_equal(x, y)
    assert x.dtype == y.dtype


def test_arange_negative():
    with pytest.raises(ValueError):
        num.arange(-10)  ###np.arange(-10) returns [] successfully
    with pytest.raises(ValueError):
        num.arange(2, -10)  ###np.arange(2, -10) returns [] successfully

    with pytest.raises(OverflowError):
        num.arange(0, num.inf)
    with pytest.raises(ValueError):
        num.arange(0, 1, num.nan)

    with pytest.raises(ZeroDivisionError):
        num.arange(0, 10, 0)
    with pytest.raises(ZeroDivisionError):
        num.arange(0.0, 10.0, 0.0)
    with pytest.raises(ZeroDivisionError):
        num.arange(0, 0, 0)
    with pytest.raises(ZeroDivisionError):
        num.arange(0.0, 0.0, 0.0)


def test_zero_with_nd_ndarray_shape():
    shape = num.array([2, 3, 4])
    x = num.zeros(shape)
    y = np.zeros(shape)
    assert np.array_equal(x, y)

    shape = np.array([2, 3, 4])
    x = num.zeros(shape)
    y = np.zeros(shape)
    assert np.array_equal(x, y)


def test_zero_with_0d_ndarray_shape():
    shape = num.array(3)
    x = num.zeros(shape)
    y = np.zeros(shape)
    assert np.array_equal(x, y)

    shape = np.array(3)
    x = num.zeros(shape)
    y = np.zeros(shape)
    assert np.array_equal(x, y)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
