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

import math

import numpy as np
import pytest

import cunumeric as num

# cunumeric.split(a: ndarray, indices: Union[int, ndarray], axis: int = 0)
# → list[cunumeric.array.ndarray]
# cunumeric.vsplit(a: ndarray, indices: Union[int, ndarray])
# → list[cunumeric.array.ndarray]    (axis=0)
# cunumeric.hsplit(a: ndarray, indices: Union[int, ndarray])
# → list[cunumeric.array.ndarray]    (axis=1)
# cunumeric.dsplit(a: ndarray, indices: Union[int, ndarray])
# → list[cunumeric.array.ndarray]    (axis=2)


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

SIZES_NO_EMPTY = [
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

ARG_FUNCS = ("vsplit", "hsplit", "dsplit")
AXIS_FUNCS = {"vsplit": 0, "hsplit": 1, "dsplit": 2}


class TestSplitErrors:
    """
    this class is to test negative cases
    """

    @pytest.mark.xfail
    def test_array_none(self):
        expected_exc = AttributeError
        with pytest.raises(expected_exc):
            np.split(None, 1)
            # Numpy raises
            # AttributeError: 'NoneType' object has no attribute 'shape'
        with pytest.raises(expected_exc):
            num.split(None, 1)
            # cuNumeric raises
            # ValueError: array(()) has less dimensions than axis(0)

    @pytest.mark.parametrize("indices", (-2, 0, "hi", 1.0, None))
    def test_indices_negative(self, indices):
        ary = num.arange(10)
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.split(ary, indices)
        with pytest.raises(expected_exc):
            np.split(ary, indices)

    def test_indices_divison(self):
        size = (3, 3, 3)
        ary = num.random.randint(low=0, high=100, size=size)
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.split(ary, 2, 0)
        with pytest.raises(expected_exc):
            np.split(ary, 2, 0)

    def test_axis_negative(self):
        ary = num.arange(10)
        expected_exc = IndexError
        axis = -2
        with pytest.raises(expected_exc):
            num.split(ary, 1, axis=axis)
        with pytest.raises(expected_exc):
            np.split(ary, 1, axis=axis)

    def test_axis_bigger(self):
        ary = num.arange(10)
        axis = 2
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.split(ary, 5, axis=axis)
        with pytest.raises(expected_exc):
            np.split(ary, 5, axis=axis)

    @pytest.mark.parametrize("func_name", ARG_FUNCS)
    def test_array_none_different_split(self, func_name):
        expected_exc = ValueError
        func_num = getattr(num, func_name)
        func_np = getattr(np, func_name)
        with pytest.raises(expected_exc):
            func_np(None, 1)
        with pytest.raises(expected_exc):
            func_num(None, 1)

    @pytest.mark.parametrize("indices", (-2, 0, "hi", 1.0, None))
    @pytest.mark.parametrize("func_name", ARG_FUNCS)
    def test_indices_negative_different_split(self, func_name, indices):
        ary = num.arange(10)
        func_num = getattr(num, func_name)
        func_np = getattr(np, func_name)

        expected_exc = ValueError
        with pytest.raises(expected_exc):
            func_num(ary, indices)
        with pytest.raises(expected_exc):
            func_np(ary, indices)

    @pytest.mark.xfail
    def test_dimensions_vsplit(self):
        ary = []
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.vsplit(ary, 1)
            # cuNumeric returns [array([], dtype=float64)]
        with pytest.raises(expected_exc):
            np.vsplit(ary, 1)
            # Numpy raises
            # ValueError: vsplit only works on arrays of 2 or more dimensions

    @pytest.mark.xfail
    def test_dimensions_vsplit_1(self):
        ary = np.random.randint(low=0, high=100, size=(5))
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.vsplit(ary, 1)
            # cuNumeric returns the array
        with pytest.raises(expected_exc):
            np.vsplit(ary, 1)
            # Numpy raises
            # ValueError: vsplit only works on arrays of 2 or more dimensions

    @pytest.mark.xfail
    def test_dimensions_hsplit_0(self):
        ary = np.random.randint(low=0, high=100, size=(0,))
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.hsplit(ary, 1)
        with pytest.raises(expected_exc):
            np.hsplit(ary, 1)
            # Numpy returns  [array([], dtype=int64)]

    def test_dimensions_hsplit_1(self):
        ary = num.arange(10)
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.hsplit(ary, 1)
        with pytest.raises(expected_exc):
            np.hsplit(ary, 1)

    @pytest.mark.parametrize("size", ((0,), (10,), (5, 5)))
    def test_dimensions_dsplit(self, size):
        ary = np.random.randint(low=0, high=100, size=size)
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.dsplit(ary, 1)
        with pytest.raises(expected_exc):
            np.dsplit(ary, 1)


def compare_array(a, b):
    """
    Compare two array using zip method.
    """
    if len(a) != len(b):
        return False
    else:
        for each in zip(a, b):
            if not np.array_equal(*each):
                return False
    return True


def get_indices(arr, axis):
    indices_arr = []

    if arr.shape[axis] == 1:
        indices_arr.append(1)  # in index

    elif arr.shape[axis] > 1:
        for div in range(2, (int)(math.sqrt(arr.shape[axis]) + 1)):
            if arr.shape[axis] % div == 0:
                indices_arr.append(div)

                # in index
                indices_arr.append(list(range(1, arr.shape[axis], div)))

                # out index
                indices_arr.append(
                    list(
                        range(
                            0,
                            arr.shape[axis] + div * np.random.randint(1, 10),
                            div,
                        )
                    )
                )
                indices_arr.append(
                    list(
                        range(
                            arr.shape[axis] + div * np.random.randint(1, 10),
                            0,
                            -div,
                        )
                    )
                )

        # if indivisible
        if len(indices_arr) == 0:
            indices_arr.append(1)
            indices_arr.append(arr.shape[axis])
            indices_arr.append(list(range(1, arr.shape[axis], 1)))

            # out index
            indices_arr.append(
                list(
                    range(0, arr.shape[axis] + 1 * np.random.randint(1, 10), 1)
                )
            )
            indices_arr.append(
                list(
                    range(
                        arr.shape[axis] + 1 * np.random.randint(1, 10), 0, -1
                    )
                )
            )

    return indices_arr


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_split(size):
    a = np.random.randint(low=0, high=100, size=size)
    axis_list = list(range(a.ndim))
    axis_list.append(-1)
    for axis in axis_list:
        input_arr = get_indices(a, axis)

        for input_opt in input_arr:
            func_num = getattr(num, "split")
            func_np = getattr(np, "split")

            res_num = func_num(a, input_opt, axis=axis)
            res_np = func_np(a, input_opt, axis=axis)

            assert compare_array(res_num, res_np)


@pytest.mark.parametrize("func_name", ARG_FUNCS)
@pytest.mark.parametrize("size", SIZES_NO_EMPTY, ids=str)
def test_split_different_split(size, func_name):
    a = np.random.randint(low=0, high=100, size=size)
    axis = AXIS_FUNCS[func_name]

    if axis >= a.ndim:
        return

    input_arr = get_indices(a, axis)
    for input_opt in input_arr:
        func_num = getattr(num, func_name)
        func_np = getattr(np, func_name)

        res_num = func_num(a, input_opt)
        res_np = func_np(a, input_opt)

        assert compare_array(res_num, res_np)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
