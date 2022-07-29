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
from utils.generators import mk_seq_array

import cunumeric as num

x = mk_seq_array(np, (3, 4, 5))
x_num = mk_seq_array(num, (3, 4, 5))
indices = mk_seq_array(np, (8,))
indices_num = num.array(indices)
indices2 = mk_seq_array(np, (3,))
indices2_num = num.array(indices2)


def test_no_axis():
    res = np.take(x, indices)
    res_num = num.take(x_num, indices_num)

    assert np.array_equal(res_num, res)


@pytest.mark.parametrize("mode", ("clip", "wrap"))
@pytest.mark.parametrize("axis", (0, 1, 2))
def test_different_axis_mode(axis, mode):
    res = np.take(x, indices, axis=axis, mode=mode)
    res_num = num.take(x_num, indices_num, axis=axis, mode=mode)
    assert np.array_equal(res_num, res)


def test_different_axis_default_mode():
    res = np.take(x, indices2, axis=1)
    res_num = num.take(x_num, indices2_num, axis=1)

    assert np.array_equal(res_num, res)


def test_different_axis_raise_mode():
    res = np.take(x, indices2, axis=2, mode="raise")
    res_num = num.take(x_num, indices2_num, axis=2, mode="raise")
    assert np.array_equal(res_num, res)


def test_with_out_array():
    out = np.ones((3, 4, 3), dtype=int)
    out_num = num.array(out)
    np.take(x, indices2, axis=2, mode="raise", out=out)
    num.take(x_num, indices2_num, axis=2, mode="raise", out=out_num)
    assert np.array_equal(out_num, out)


def test_indices_list_single():
    x = np.arange(6)
    x_num = num.array(x)

    res = x.take([3], axis=0)
    res_num = x_num.take([3], axis=0)

    assert np.array_equal(res_num, res)


def test_indices_list():
    x = np.arange(6)
    x_num = num.array(x)

    res = x.take([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], axis=0)
    res_num = x_num.take([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], axis=0)

    assert np.array_equal(res_num, res)


@pytest.mark.parametrize("mode", ("clip", "wrap"))
def test_scalar_indices_mode(mode):
    res = np.take(x, 7, axis=0, mode=mode)
    res_num = num.take(x_num, 7, axis=0, mode=mode)

    assert np.array_equal(res_num, res)


@pytest.mark.parametrize("mode", ("clip", "wrap"))
@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_ndim(ndim, mode):
    shape = (5,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    np_indices = mk_seq_array(np, (4,))
    num_indices = mk_seq_array(num, (4,))
    res_np = np.take(np_arr, np_indices)
    res_num = num.take(num_arr, num_indices)

    assert np.array_equal(res_num, res_np)
    for axis in range(ndim):
        res_np = np.take(np_arr, np_indices, axis=axis)
        res_num = num.take(num_arr, num_indices, axis=axis)
        assert np.array_equal(res_num, res_np)

    np_indices = mk_seq_array(np, (8,))
    num_indices = mk_seq_array(num, (8,))

    res_np = np.take(np_arr, np_indices, mode=mode)
    res_num = num.take(num_arr, num_indices, mode=mode)

    assert np.array_equal(res_num, res_np)
    for axis in range(ndim):
        res_np = np.take(np_arr, np_indices, axis=axis, mode=mode)
        res_num = num.take(num_arr, num_indices, axis=axis, mode=mode)
        assert np.array_equal(res_num, res_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
