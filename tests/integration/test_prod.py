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

# numpy.prod(a, axis=None, dtype=None, out=None, keepdims=<no value>,
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
SIZES_E2 = [
    (DIM),
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, 1, DIM),
    (DIM, DIM, DIM),
]
SIZE_E = [
    (1, 1),
    (1, DIM),
    (DIM, 1),
    (1, 1, 1),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, 1, DIM),
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
COMPLEX_TYPE = ["F"]
NEGATIVE_COMPLEX_TYPE = ["D"]
NEGATIVE_DTYPE = ["h", "i", "H", "I", "?", "b", "B"]


def to_dtype(s):
    return str(np.dtype(s))


class TestProdNegative(object):
    """
    this class is to test negative cases
    """

    @pytest.mark.parametrize("arr", ARR)
    def test_array(self, arr):
        assert np.array_equal(np.prod(arr), num.prod(arr))

    @pytest.mark.xfail
    @pytest.mark.parametrize("dtype", NEGATIVE_DTYPE, ids=to_dtype)
    def test_dtype_negative(self, dtype):
        size = (5, 5, 5)
        arr = np.random.random(size) * 10 + 2
        arr_np = np.array(arr, dtype=dtype)
        arr_num = num.array(arr_np)
        out_np = np.prod(arr_np)  # Numpy return product of all datas
        out_num = num.prod(arr_num)
        # cuNumeric return an array with a different data
        assert allclose(out_np, out_num)

    @pytest.mark.skip
    @pytest.mark.parametrize("dtype", NEGATIVE_COMPLEX_TYPE, ids=to_dtype)
    def test_dtype_complex_negative(self, dtype):
        arr = (num.random.rand(5, 5) * 10 + 2) + (
            num.random.rand(5, 5) * 10 * 1.0j + 0.2j
        )
        arr_np = np.array(arr, dtype=dtype)
        arr_num = num.array(arr_np)
        out_np = np.prod(arr_np)
        out_num = num.prod(arr_num)
        assert allclose(out_np, out_num)

    def test_axis_out_bound(self):
        arr = [-1, 0, 1, 2, 10]
        msg = r"bounds"
        with pytest.raises(np.AxisError, match=msg):
            num.prod(arr, axis=2)

    @pytest.mark.xfail
    @pytest.mark.parametrize("axis", ((-1, 1), (0, 1), (1, 2), (0, 2)))
    def test_axis_tuple(self, axis):
        size = (5, 5, 5)
        arr_np = np.random.random(size) * 10
        arr_num = num.array(arr_np)
        out_np = np.prod(arr_np, axis=axis)
        # cuNumeric raises NotImplementedError:
        # Need support for reducing multiple dimensions.
        # Numpy get results.
        out_num = num.prod(arr_num, axis=axis)
        assert allclose(out_np, out_num)

    def test_out_negative(self):
        in_shape = (2, 3, 4)
        out_shape = (2, 3, 3)
        arr_num = num.random.random(in_shape) * 10
        arr_out = num.random.random(out_shape) * 10
        msg = r"shapes do not match"
        with pytest.raises(ValueError, match=msg):
            num.prod(arr_num, out=arr_out, axis=2)

    def test_keepdims(self):
        in_shape = (2, 3, 4)
        arr_num = num.random.random(in_shape) * 10
        arr_np = np.array(arr_num)
        out_np = np.prod(arr_np, axis=2, keepdims=True)
        out_num = num.prod(arr_num, axis=2, keepdims=True)
        assert np.array_equal(out_np, out_num)

    @pytest.mark.xfail
    def test_initial_scalar_list(self):
        arr = [[1, 2], [3, 4]]
        initial_value = [3]

        out_num = num.prod(arr, initial=initial_value)  # array(72)
        # Numpy raises ValueError:
        # Input object to FillWithScalar is not a scalar
        out_np = np.prod(arr, initial=initial_value)

        assert np.array_equal(out_np, out_num)

    def test_initial_list(self):
        arr = [[1, 2], [3, 4]]
        initial_value = [2, 3]
        with pytest.raises(ValueError):
            num.prod(arr, initial=initial_value)

    def test_initial_empty_array(self):
        size = (1, 0)
        arr_np = np.random.random(size) * 10
        arr_num = num.array(arr_np)
        initial_value = random.uniform(-20.0, 20.0)
        out_num = num.prod(arr_num, initial=initial_value)
        out_np = np.prod(arr_np, initial=initial_value)
        assert allclose(out_np, out_num)

    @pytest.mark.xfail
    def test_where(self):
        arr = [[1, 2], [3, 4]]
        out_np = np.prod(arr, where=[False, True])  # return 8
        # cuNumeric raises NotImplementedError:
        # the `where` parameter is currently not supported
        out_num = num.prod(arr, where=[False, True])
        assert np.array_equal(out_np, out_num)


class TestProdPositive(object):
    """
    this class is to test positive cases
    """

    @pytest.mark.parametrize("size", SIZES)
    def test_basic(self, size):
        arr_np = np.random.random(size)
        arr_num = num.array(arr_np)
        out_np = np.prod(arr_np)
        out_num = np.prod(arr_num)
        assert allclose(out_np, out_num)

    @pytest.mark.parametrize("dtype", DTYPE, ids=to_dtype)
    def test_dtype(self, dtype):
        size = (5, 5, 5)
        arr = np.random.random(size) * 10
        arr_np = np.array(arr, dtype=dtype)
        arr_num = num.array(arr_np)
        out_np = np.prod(arr_np)
        out_num = num.prod(arr_num)
        assert allclose(out_np, out_num)

    @pytest.mark.parametrize("dtype", COMPLEX_TYPE, ids=to_dtype)
    def test_dtype_complex(self, dtype):
        arr = (num.random.rand(5, 5) * 10 + 2) + (
            num.random.rand(5, 5) * 10 * 1.0j + 0.2j
        )
        arr_np = np.array(arr, dtype=dtype)
        arr_num = num.array(arr_np)
        out_np = np.prod(arr_np)
        out_num = num.prod(arr_num)
        assert allclose(out_np, out_num)

    @pytest.mark.parametrize("axis", (_ for _ in range(-2, 3, 1)))
    def test_axis_basic(self, axis):
        size = (5, 5, 5)
        arr_np = np.random.random(size) * 10
        arr_num = num.array(arr_np)
        out_num = num.prod(arr_num, axis=axis)
        out_np = np.prod(arr_np, axis=axis)
        assert allclose(out_np, out_num)

    @pytest.mark.parametrize("size", SIZES)
    def test_out_basic(self, size):
        arr_np = np.random.random(size)
        arr_num = num.array(arr_np)
        out_np = np.random.random(())
        out_num = num.random.random(())
        np.prod(arr_np, out=out_np)
        num.prod(arr_num, out=out_num)
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
            np.prod(arr_np, out=out_np, axis=axis)
            num.prod(arr_num, out=out_num, axis=axis)
            assert allclose(out_np, out_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize("size", SIZES_E2)
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

            np.prod(arr_np, out=out_np, axis=axis)
            num.prod(arr_num, out=out_num, axis=axis)

            assert allclose(out_np, out_num)

    @pytest.mark.parametrize("size", SIZES)
    def test_axis_keepdims_false(self, size):
        arr_np = np.random.random(size)
        arr_num = num.array(arr_np)
        ndim = arr_np.ndim
        for axis in range(-ndim + 1, ndim, 1):
            out_np = np.prod(arr_np, axis=axis, keepdims=False)
            out_num = num.prod(arr_num, axis=axis, keepdims=False)
            assert allclose(out_np, out_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize("size", SIZE_E)
    def test_axis_keepdims_true(self, size):
        arr_np = np.random.random(size)
        arr_num = num.array(arr_np)
        ndim = arr_np.ndim
        for axis in range(-ndim + 1, ndim, 1):
            out_np = np.prod(arr_np, axis=axis, keepdims=True)
            out_num = num.prod(arr_num, axis=axis, keepdims=True)
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
        out_num = num.prod(arr_num, initial=initial_value)
        out_np = np.prod(arr_np, initial=initial_value)

        assert allclose(out_np, out_num)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
