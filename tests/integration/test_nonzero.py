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

import numpy as np
import pytest

import cunumeric as num

# cunumeric.count_nonzero(a: ndarray,
# axis: Optional[Union[int, tuple[int, ...]]] = None) → Union[int, ndarray]
# cunumeric.nonzero(a: ndarray) → tuple[cunumeric.array.ndarray, ...]
# cunumeric.flatnonzero(a: ndarray) → ndarray

DIM = 5
EMPTY_SIZES = [
    (0,),
    (0, 1),
    (1, 0),
    (1, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
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

SIZES = NO_EMPTY_SIZE + EMPTY_SIZES


@pytest.mark.parametrize("size", EMPTY_SIZES)
def test_empty(size):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    res_np = np.count_nonzero(arr_np)
    res_num = num.count_nonzero(arr_num)
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("arr", ([], [[], []], [[[], []], [[], []]]))
def test_empty_arr(arr):
    res_np = np.count_nonzero(arr)
    res_num = num.count_nonzero(arr)
    assert np.array_equal(res_np, res_num)


def assert_equal(numarr, nparr):
    for resultnp, resultnum in zip(nparr, numarr):
        assert np.array_equal(resultnp, resultnum)


@pytest.mark.parametrize("size", NO_EMPTY_SIZE)
def test_basic(size):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    res_np = np.count_nonzero(arr_np)
    res_num = num.count_nonzero(arr_num)
    np.array_equal(res_np, res_num)


def test_axis_out_bound():
    arr = [-1, 0, 1, 2, 10]
    with pytest.raises(np.AxisError):
        num.count_nonzero(arr, axis=2)


@pytest.mark.xfail
@pytest.mark.parametrize("axis", ((-1, 1), (0, 1), (1, 2), (0, 2)))
def test_axis_tuple(axis):
    size = (5, 5, 5)
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    out_np = np.count_nonzero(arr_np, axis=axis)
    # Numpy passed all axis values
    out_num = num.count_nonzero(arr_num, axis=axis)
    # For (-1, 1), cuNumeric raises 'ValueError:
    # Invalid promotion on dimension 2 for a 1-D store'
    # For the others, cuNumeric raises 'NotImplementedError:
    # Need support for reducing multiple dimensions'
    assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize("size", NO_EMPTY_SIZE)
def test_basic_axis(size):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    ndim = arr_np.ndim
    for axis in range(-ndim + 1, ndim, 1):
        out_np = np.count_nonzero(arr_np, axis=axis)
        out_num = num.count_nonzero(arr_num, axis=axis)
        assert np.array_equal(out_np, out_num)


@pytest.mark.xfail
@pytest.mark.parametrize("size", EMPTY_SIZES)
def test_empty_axis(size):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    ndim = arr_np.ndim
    for axis in range(-ndim + 1, ndim, 1):
        out_np = np.count_nonzero(arr_np, axis=axis)
        out_num = num.count_nonzero(arr_num, axis=axis)
        # Numpy and cuNumeric have diffrent out.
        # out_np = array([[0]])
        # out_num = 0
        assert np.array_equal(out_np, out_num)


@pytest.mark.xfail
@pytest.mark.parametrize("size", NO_EMPTY_SIZE)
@pytest.mark.parametrize("keepdims", [False, True])
def test_axis_keepdims(size, keepdims):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    ndim = arr_np.ndim
    for axis in range(-ndim + 1, ndim, 1):
        out_np = np.count_nonzero(arr_np, axis=axis, keepdims=keepdims)
        out_num = num.count_nonzero(arr_num, axis=axis, keepdims=keepdims)
        # Numpy has the parameter 'keepdims',
        # cuNumeric do not have this parameter.
        # cuNumeric raises "TypeError: count_nonzero() got an unexpected
        # keyword argument 'keepdims'"
        assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize("size", SIZES)
def test_nonzero(size):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    res_np = np.nonzero(arr_np)
    res_num = num.nonzero(arr_num)
    np.array_equal(res_np, res_num)


@pytest.mark.parametrize("size", SIZES)
def test_flatnonzero(size):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    res_np = np.flatnonzero(arr_np)
    res_num = num.flatnonzero(arr_num)
    np.array_equal(res_np, res_num)


def test_deprecated_0d():
    with pytest.deprecated_call():
        assert num.count_nonzero(num.array(0)) == 0
        assert num.count_nonzero(num.array(0, dtype="?")) == 0
        assert_equal(num.nonzero(0), np.nonzero(0))

    with pytest.deprecated_call():
        assert num.count_nonzero(num.array(1)) == 1
        assert num.count_nonzero(num.array(1, dtype="?")) == 1
        assert_equal(num.nonzero(1), np.nonzero(1))

    with pytest.deprecated_call():
        assert_equal(num.nonzero(0), ([],))

    with pytest.deprecated_call():
        assert_equal(num.nonzero(1), ([0],))

    x_np = np.array([True, True])
    x = num.array(x_np)
    assert np.array_equal(x_np.nonzero(), x.nonzero())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
