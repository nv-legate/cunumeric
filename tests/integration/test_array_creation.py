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


CREATION_FUNCTIONS = ("zeros", "ones")
FILLED_VALUES = [0, 1, 1000, 123.456]
SIZES = (0, 1, 2)
NDIMS = 5
DTYPES = (np.uint32, np.int32, np.float64, np.complex128)


def test_empty():
    par = (SIZES, range(NDIMS), DTYPES)
    for size, ndims, dtype in product(*par):
        shape = ndims * [size]

        xf = num.empty(shape, dtype=dtype)
        yf = np.empty(shape, dtype=dtype)

        assert xf.shape == yf.shape
        assert xf.dtype == yf.dtype


@pytest.mark.parametrize("fn", CREATION_FUNCTIONS)
def test_creation_func(fn):
    num_f = getattr(num, fn)
    np_f = getattr(np, fn)

    par = (SIZES, range(NDIMS), DTYPES)
    for size, ndims, dtype in product(*par):
        shape = ndims * [size]

        xf = num_f(shape, dtype=dtype)
        yf = np_f(shape, dtype=dtype)

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
    np.array([2, -3, 4]),
]


class TestCreationErrors:
    def setup_method(self):
        self.bad_type_shape = (2, 3.0)

    @pytest.mark.parametrize("shape", SHAPES_NEGATIVE, ids=str)
    class TestNegativeShape:
        @pytest.mark.parametrize("fn", ("empty", "zeros", "ones"))
        def test_creation(self, shape, fn):
            with pytest.raises(ValueError):
                getattr(num, fn)(shape)

        def test_full(self, shape):
            with pytest.raises(ValueError):
                num.full(shape, 10)

    @pytest.mark.parametrize("fn", ("empty", "zeros", "ones"))
    def test_creation_bad_type(self, fn):
        with pytest.raises(TypeError):
            getattr(num, fn)(self.bad_type_shape)

    def test_full_bad_type(self):
        with pytest.raises(TypeError):
            num.full(self.bad_type_shape, 10)

    # additional special case for full
    def test_full_bad_filled_value(self):
        with pytest.raises(ValueError):
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
    (np.array([[[]]]), None),
    (np.arange(24).reshape(2, 3, 4), None),
    (np.arange(24).reshape(4, 3, 2), "f4"),
]
LIKE_FUNCTIONS = ("zeros_like", "ones_like")
SHAPE_ARG = (None, (-1,), (1, -1))


@pytest.mark.parametrize("x_np,dtype", DATA_ARGS)
@pytest.mark.parametrize("shape", SHAPE_ARG)
def test_empty_like(x_np, dtype, shape):
    shape = shape if shape is None else x_np.reshape(shape).shape
    x = num.array(x_np)
    xfl = num.empty_like(x, dtype=dtype, shape=shape)
    yfl = np.empty_like(x_np, dtype=dtype, shape=shape)

    assert xfl.shape == yfl.shape
    assert xfl.dtype == yfl.dtype


@pytest.mark.parametrize("x_np,dtype", DATA_ARGS)
@pytest.mark.parametrize("fn", LIKE_FUNCTIONS)
@pytest.mark.parametrize("shape", SHAPE_ARG)
def test_func_like(fn, x_np, dtype, shape):
    shape = shape if shape is None else x_np.reshape(shape).shape
    num_f = getattr(num, fn)
    np_f = getattr(np, fn)

    x = num.array(x_np)
    xfl = num_f(x, dtype=dtype, shape=shape)
    yfl = np_f(x_np, dtype=dtype, shape=shape)

    assert np.array_equal(xfl, yfl)
    assert xfl.dtype == yfl.dtype


@pytest.mark.parametrize("value", FILLED_VALUES)
@pytest.mark.parametrize("x_np, dtype", DATA_ARGS)
@pytest.mark.parametrize("shape", SHAPE_ARG)
def test_full_like(x_np, dtype, value, shape):
    shape = shape if shape is None else x_np.reshape(shape).shape
    x = num.array(x_np)

    xfl = num.full_like(x, value, dtype=dtype, shape=shape)
    yfl = np.full_like(x_np, value, dtype=dtype, shape=shape)
    assert np.array_equal(xfl, yfl)
    assert xfl.dtype == yfl.dtype


def test_full_like_bad_filled_value():
    x = num.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        num.full_like(x, [10, 20, 30])


ARANGE_ARGS = [
    (0,),
    (10,),
    (3.5,),
    pytest.param((-10), marks=pytest.mark.xfail),
    (2, 10),
    pytest.param((2, -10), marks=pytest.mark.xfail),
    (-2.5, 10.0),
    pytest.param((1, -10, -2.5), marks=pytest.mark.xfail),
    (1.0, -10.0, -2.5),
    (-10, 10, 10),
    (-10, 10, -100),
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


ARANGE_ARGS_STEP_ZERO = [
    (0, 0, 0),
    (0, 10, 0),
    (-10, 10, 0),
    (1, 10, 0),
    (10, -10, 0),
    (0.0, 0.0, 0.0),
    (0.0, 10.0, 0.0),
    (-10.0, 10.0, 0.0),
    (1.0, 10.0, 0.0),
    (10.0, -10.0, 0.0),
]


class TestArrangeErrors:
    def test_inf(self):
        with pytest.raises(OverflowError):
            num.arange(0, num.inf)

    def test_nan(self):
        with pytest.raises(ValueError):
            num.arange(0, 1, num.nan)

    @pytest.mark.parametrize("args", ARANGE_ARGS_STEP_ZERO, ids=str)
    def test_zero_division(self, args):
        with pytest.raises(ZeroDivisionError):
            num.arange(*args)


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

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
