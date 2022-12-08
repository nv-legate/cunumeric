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
import random

import numpy as np
import pytest
from utils.comparisons import allclose

import cunumeric as num

# numpy.sum(a, axis=None, dtype=None, out=None, keepdims=<no value>,
# initial=<no value>, where=<no value>)

DIM = 5
SIZES = [
    (0,),
    (1),
    (DIM),
    (0, 1),
    (1, 0),
    (1, 1),
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
    (1, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (1, 1, 1),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, 1, DIM),
    (DIM, DIM, DIM),
]

NO_EMPTY_SIZE = [
    (1),
    (DIM),
    (1, 1),
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
    (1, 1, 1),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, 1, DIM),
    (DIM, DIM, DIM),
]

ARR = ([], [[]], [[], []], np.inf, np.Inf, -10.3, 0, 200, 5 + 8j)

DTYPE = ["l", "L", "f", "e", "d"]
COMPLEX_TYPE = ["F", "D"]
NEGATIVE_DTYPE = ["h", "i", "H", "I", "?", "b", "B"]


def to_dtype(s):
    return str(np.dtype(s))


class TestSumNegative(object):
    """
    this class is to test negative cases
    """

    def test_array_none(self):
        assert num.sum(None) is None

    @pytest.mark.parametrize("arr", ARR)
    def test_array(self, arr):
        assert np.array_equal(np.sum(arr), num.sum(arr))

    @pytest.mark.xfail
    @pytest.mark.parametrize("dtype", NEGATIVE_DTYPE, ids=to_dtype)
    def test_dtype_negative(self, dtype):
        size = (5, 5, 5)
        arr = np.random.random(size) * 10
        arr_np = np.array(arr, dtype=dtype)
        arr_num = num.array(arr_np)
        out_np = np.sum(arr_np)  # Numpy return sum of all datas
        out_num = num.sum(
            arr_num
        )  # cuNumeric return an array with different data
        assert allclose(out_np, out_num)

    def test_axis_out_bound(self):
        arr = [-1, 0, 1, 2, 10]
        msg = r"bounds"
        with pytest.raises(np.AxisError, match=msg):
            num.sum(arr, axis=2)

    @pytest.mark.xfail
    @pytest.mark.parametrize("axis", ((-1, 1), (0, 1), (1, 2), (0, 2)))
    def test_axis_tuple(self, axis):
        size = (5, 5, 5)
        arr_np = np.random.random(size) * 10
        arr_num = num.array(arr_np)
        out_np = np.sum(arr_np, axis=axis)
        # cuNumeric raises NotImplementedError:
        # 'Need support for reducing multiple dimensions'
        # Numpy get results
        out_num = num.sum(arr_num, axis=axis)
        assert allclose(out_np, out_num)

    def test_out_negative(self):
        in_shape = (2, 3, 4)
        out_shape = (2, 3, 3)
        arr_num = num.random.random(in_shape) * 10
        arr_out = num.random.random(out_shape) * 10
        msg = r"shapes do not match"
        with pytest.raises(ValueError, match=msg):
            num.sum(arr_num, out=arr_out, axis=2)

    def test_keepdims(self):
        in_shape = (2, 3, 4)
        arr_num = num.random.random(in_shape) * 10
        arr_np = np.array(arr_num)
        out_np = np.sum(arr_np, axis=2, keepdims=True)
        out_num = num.sum(arr_num, axis=2, keepdims=True)
        assert np.array_equal(out_np, out_num)

    @pytest.mark.xfail
    def test_initial_scalar_list(self):
        arr = [[1, 2], [3, 4]]
        initial_value = [3]
        out_num = num.sum(arr, initial=initial_value)  # array(13)
        out_np = np.sum(
            arr, initial=initial_value
        )  # ValueError: Input object to FillWithScalar is not a scalar
        assert np.array_equal(out_np, out_num)

    def test_initial_list(self):
        arr = [[1, 2], [3, 4]]
        initial_value = [2, 3]
        with pytest.raises(ValueError):
            num.sum(arr, initial=initial_value)

    @pytest.mark.xfail
    def test_initial_empty_array(self):
        size = (1, 0)
        arr_np = np.random.random(size) * 10
        arr_num = num.array(arr_np)
        initial_value = random.uniform(-20.0, 20.0)
        out_num = num.sum(arr_num, initial=initial_value)  # return 0.0
        out_np = np.sum(arr_np, initial=initial_value)  # return initial_value
        assert allclose(out_np, out_num)

    @pytest.mark.xfail
    def test_where(self):
        arr = [[1, 2], [3, 4]]
        out_np = np.sum(arr, where=[False, True])  # return 6
        # cuNumeric raises NotImplementedError:
        # "the `where` parameter is currently not supported"
        out_num = num.sum(arr, where=[False, True])
        assert np.array_equal(out_np, out_num)


class TestSumPositive(object):
    """
    this class is to test positive cases
    """

    @pytest.mark.parametrize("size", SIZES)
    def test_basic(self, size):
        arr_np = np.random.random(size)
        arr_num = num.array(arr_np)
        out_np = np.sum(arr_np)
        out_num = np.sum(arr_num)
        assert allclose(out_np, out_num)

    @pytest.mark.parametrize("dtype", DTYPE, ids=to_dtype)
    def test_dtype(self, dtype):
        size = (5, 5, 5)
        arr = np.random.random(size) * 10
        arr_np = np.array(arr, dtype=dtype)
        arr_num = num.array(arr_np)
        out_np = np.sum(arr_np)
        out_num = num.sum(arr_num)
        assert allclose(out_np, out_num)

    @pytest.mark.parametrize("dtype", COMPLEX_TYPE, ids=to_dtype)
    def test_dtype_complex(self, dtype):
        arr = num.random.rand(5, 5) * 10 + num.random.rand(5, 5) * 10 * 1.0j
        arr_np = np.array(arr, dtype=dtype)
        arr_num = num.array(arr_np)
        out_np = np.sum(arr_np)
        out_num = num.sum(arr_num)
        assert allclose(out_np, out_num)

    @pytest.mark.parametrize("axis", (_ for _ in range(-2, 3, 1)))
    def test_axis_basic(self, axis):
        size = (5, 5, 5)
        arr_np = np.random.random(size) * 10
        arr_num = num.array(arr_np)
        out_num = num.sum(arr_num, axis=axis)
        out_np = np.sum(arr_np, axis=axis)
        assert allclose(out_np, out_num)

    @pytest.mark.parametrize("size", SIZES)
    def test_out_basic(self, size):
        arr_np = np.random.random(size)
        arr_num = num.array(arr_np)
        out_np = np.random.random(())
        out_num = num.random.random(())
        np.sum(arr_np, out=out_np)
        num.sum(arr_num, out=out_num)
        assert allclose(out_np, out_num)

    @pytest.mark.parametrize("size", SIZES)
    def test_out_axis(self, size):
        arr_np = np.random.random(size)
        arr_num = num.array(arr_np)
        ndim = arr_np.ndim
        for axis in range(-ndim + 1, ndim, 1):
            out_shape = ()
            if type(size) == tuple:
                out_shape_list = list(size)
                del out_shape_list[axis]
                out_shape = tuple(out_shape_list)
            out_np = np.random.random(out_shape)
            out_num = num.random.random(out_shape)
            np.sum(arr_np, out=out_np, axis=axis)
            num.sum(arr_num, out=out_num, axis=axis)
            assert allclose(out_np, out_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize("size", SIZES)
    def test_out_axis_dtype(self, size):
        arr = np.random.random(size) * 10
        arr_np = np.array(arr, dtype=to_dtype("f"))
        arr_num = num.array(arr, dtype=to_dtype("f"))

        ndim = arr_np.ndim
        for axis in range(-ndim + 1, ndim, 1):
            out_shape = ()
            if type(size) == tuple:
                out_shape_list = list(size)
                del out_shape_list[axis]
                out_shape = tuple(out_shape_list)
            out = np.random.random(out_shape)

            out_np = np.array(out, dtype=to_dtype("i"))
            out_num = num.array(out, dtype=to_dtype("i"))

            np.sum(arr_np, out=out_np, axis=axis)
            num.sum(arr_num, out=out_num, axis=axis)

            # some data in the out_result are different
            # out_np     = array([[39, 23, 22, 37, 19],
            #        [21, 28, 29, 38, 24],
            #        [29, 25, 30, 27, 23],
            #        [24, 30, 22, 29, 22],
            #        [16, 15, 29, 22, 13]], dtype=int32)
            # out_num    = array([[38, 21, 20, 35, 17],
            #        [19, 25, 27, 37, 22],
            #        [27, 24, 29, 24, 22],
            #        [21, 27, 20, 26, 19],
            #        [13, 14, 26, 20, 12]], dtype=int32)

            assert allclose(out_np, out_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_axis_keepdims(self, size, keepdims):
        arr_np = np.random.random(size)
        arr_num = num.array(arr_np)
        ndim = arr_np.ndim
        for axis in range(-ndim + 1, ndim, 1):
            out_np = np.sum(arr_np, axis=axis, keepdims=keepdims)
            out_num = num.sum(arr_num, axis=axis, keepdims=keepdims)
            # in cunumeric/deferred/unary_reduction:
            # if lhs_array.size == 1:
            #     > assert axes is None or len(axes) == rhs_array.ndim - (
            #         0 if keepdims else lhs_array.ndim
            #     )
            # E    AssertionError
            assert allclose(out_np, out_num)

    @pytest.mark.parametrize("size", NO_EMPTY_SIZE)
    def test_initial(self, size):
        arr_np = np.random.random(size) * 10
        arr_num = num.array(arr_np)
        initial_value = random.uniform(-20.0, 20.0)
        out_num = num.sum(arr_num, initial=initial_value)
        out_np = np.sum(arr_np, initial=initial_value)

        assert allclose(out_np, out_num)


def test_indexed():
    x_np = np.random.randn(100)
    indices = np.random.choice(
        np.arange(x_np.size), replace=False, size=int(x_np.size * 0.2)
    )
    x_np[indices] = 0
    x = num.array(x_np)
    assert allclose(num.sum(x), np.sum(x_np))

    x_np = x_np.reshape(10, 10)
    x = num.array(x_np)
    assert allclose(num.sum(x), np.sum(x_np))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
