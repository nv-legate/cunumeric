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
from legate.core import LEGATE_MAX_DIM

import cunumeric as num

INPUTS = (
    [-1, 4, 5],
    [5, 10, 0, 100],
    [[0, 0], [0, 0]],
    [[True, True, False], [True, True, True]],
    [[False, True, False]],
    [[0.0, 1.0, 0.0]],
    [[1, 0 + 1j, 1 + 1j]],
    [[1, 0 + 1j, 0 + 0j]],
    [np.nan],
)

FUNCTIONS = ("all", "any")


@pytest.mark.parametrize("keepdims", (False, True))
@pytest.mark.parametrize("input", INPUTS)
@pytest.mark.parametrize("func", FUNCTIONS)
def test_basic(func, input, keepdims):
    in_np = np.array(input)
    # cuNumeric doesn't support reductions for complex128
    if in_np.dtype.kind == "c":
        in_np = in_np.astype("F")
    in_num = num.array(in_np)

    fn_np = getattr(np, func)
    fn_num = getattr(num, func)
    assert np.array_equal(
        fn_np(in_np, keepdims=keepdims), fn_num(in_num, keepdims=keepdims)
    )
    for axis in range(-in_num.ndim, in_num.ndim):
        out_np = fn_np(in_np, axis=axis, keepdims=keepdims)
        out_num = fn_num(in_num, axis=axis, keepdims=keepdims)
        assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize(
    "axes",
    ((0,), (1, 2), pytest.param((-1, 0), marks=pytest.mark.xfail), (-1, 0, 1)),
    ids=lambda axes: f"(axes={axes})",
)
@pytest.mark.parametrize("func", FUNCTIONS)
def test_axis_tuple(func, axes):
    # For axes=(-1, 0),
    # in Numpy, it pass
    # in cuNumeric, raises ValueError:
    # Invalid promotion on dimension 2 for a 1-D store
    input = [[[5, 10], [0, 100]]]
    in_np = np.array(input)
    in_num = num.array(in_np)

    fn_np = getattr(np, func)
    fn_num = getattr(num, func)
    out_np = fn_np(in_np, axis=axes)
    out_num = fn_num(in_num, axis=axes)
    assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize("ndim", range(LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize("func", FUNCTIONS)
def test_nd_inputs(ndim, func):
    shape = (3,) * ndim
    in_np = np.random.random(shape)
    in_num = num.array(in_np)

    fn_np = getattr(np, func)
    fn_num = getattr(num, func)
    for axis in range(in_num.ndim):
        out_np = fn_np(in_np, axis=axis)
        out_num = fn_num(in_num, axis=axis)
        assert np.array_equal(out_np, out_num)

        out_np = np.empty(out_np.shape, dtype="D")
        out_num = num.empty(out_num.shape, dtype="D")
        fn_np(in_np, axis=axis, out=out_np)
        fn_num(in_num, axis=axis, out=out_num)
        assert np.array_equal(out_np, out_num)

        out_np = fn_np(in_np[1:], axis=axis)
        out_num = fn_num(in_num[1:], axis=axis)
        assert np.array_equal(out_np, out_num)


@pytest.mark.skip
def test_where():
    # "the `where` parameter is currently not supported"
    x = np.array([[True, True, False], [True, True, True]])
    y = np.array([[True, False], [True, True]])
    cy = num.array(y)

    assert num.array_equal(
        num.all(cy, where=[True, False]), np.all(x, where=[True, False])
    )
    assert num.array_equal(
        num.any(cy, where=[[True], [False]]),
        np.any(x, where=[[True], [False]]),
    )


class TestAnyAllErrors:
    def setup_method(self):
        input = [[[5, 10], [0, 100]]]
        self.in_np = np.array(input)
        self.in_num = num.array(self.in_np)

    @pytest.mark.parametrize(
        "axis", (-4, 3), ids=lambda axis: f"(axis={axis})"
    )
    @pytest.mark.parametrize("func", FUNCTIONS)
    def test_axis_out_of_bound(self, func, axis):
        expected_exc = ValueError
        fn_np = getattr(np, func)
        fn_num = getattr(num, func)

        with pytest.raises(expected_exc):
            fn_np(self.in_np, axis=axis)
        with pytest.raises(expected_exc):
            fn_num(self.in_num, axis=axis)

    @pytest.mark.parametrize(
        "axes", ((1, 1), (-1, 2), (0, 3)), ids=lambda axes: f"(axes={axes})"
    )
    @pytest.mark.parametrize("func", FUNCTIONS)
    def test_invalid_axis_tuple(self, func, axes):
        expected_exc = ValueError
        fn_np = getattr(np, func)
        fn_num = getattr(num, func)

        with pytest.raises(expected_exc):
            fn_np(self.in_np, axis=axes)
        with pytest.raises(expected_exc):
            fn_num(self.in_num, axis=axes)

    @pytest.mark.parametrize(
        ("axis", "out_shape"), ((None, (1,)), (1, (2,)), (1, (2, 2)))
    )
    @pytest.mark.parametrize("func", FUNCTIONS)
    def test_out_invalid_shape(self, func, axis, out_shape):
        expected_exc = ValueError
        func_np = getattr(np, func)
        func_num = getattr(num, func)

        out_np = np.empty(out_shape)
        out_num = num.empty(out_shape)

        with pytest.raises(expected_exc):
            func_np(self.in_np, axis=axis, out=out_np)
        with pytest.raises(expected_exc):
            func_num(self.in_num, axis=axis, out=out_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
