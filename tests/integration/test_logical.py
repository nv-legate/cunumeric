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


@pytest.mark.parametrize("input", INPUTS)
def test_any_and_all(input):
    in_np = np.array(input)
    # cuNumeric doesn't support reductions for complex128
    if in_np.dtype.kind == "c":
        in_np = in_np.astype("F")
    in_num = num.array(in_np)

    for fn in ("any", "all"):
        fn_np = getattr(np, fn)
        fn_num = getattr(num, fn)
        assert np.array_equal(fn_np(in_np), fn_num(in_num))
        for axis in range(in_num.ndim):
            out_np = fn_np(in_np, axis=axis)
            out_num = fn_num(in_num, axis=axis)
            assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize("ndim", range(LEGATE_MAX_DIM + 1))
def test_nd_inputs(ndim):
    shape = (3,) * ndim
    in_np = np.random.random(shape)
    in_num = num.array(in_np)

    for fn in ("any", "all"):
        fn_np = getattr(np, fn)
        fn_num = getattr(num, fn)
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


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
