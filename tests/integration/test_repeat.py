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
from test_tools.generators import mk_seq_array

import cunumeric as num
from legate.core import LEGATE_MAX_DIM

np.random.seed(12345)


def test_basic():
    assert np.array_equal(num.repeat(3, 4), np.repeat(3, 4))
    assert np.array_equal(num.repeat([3, 1], 4), np.repeat([3, 1], 4))


def test_axis():
    anp = np.array([1, 2, 3, 4, 5])
    a = num.array(anp)
    repnp = np.array([1, 2, 1, 2, 1])
    rep = num.array(repnp)
    print(num.repeat(a, rep, axis=0))
    print(np.repeat(anp, repnp, axis=0))
    assert np.array_equal(
        num.repeat(a, rep, axis=0), np.repeat(anp, repnp, axis=0)
    )
    xnp = np.array([[1, 2], [3, 4]])
    x = num.array([[1, 2], [3, 4]])
    assert np.array_equal(
        num.repeat(x, [1, 2], axis=0), np.repeat(xnp, [1, 2], axis=0)
    )
    assert np.array_equal(num.repeat(x, 0, axis=0), np.repeat(xnp, 0, axis=0))


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_nd(ndim):
    a_shape = tuple(np.random.randint(1, 9) for _ in range(ndim))
    np_array = mk_seq_array(np, a_shape)
    num_array = mk_seq_array(num, a_shape)
    repeats = np.random.randint(0, 15)
    res_num = num.repeat(num_array, repeats)
    res_np = np.repeat(np_array, repeats)
    assert np.array_equal(res_num, res_np)
    for axis in range(0, ndim):
        res_num2 = num.repeat(num_array, repeats, axis)
        res_np2 = np.repeat(np_array, repeats, axis)
        assert np.array_equal(res_num2, res_np2)
        rep_shape = (a_shape[axis],)
        rep_arr_np = mk_seq_array(np, rep_shape)
        rep_arr_num = mk_seq_array(num, rep_shape)
        res_num3 = num.repeat(num_array, rep_arr_num, axis)
        res_np3 = np.repeat(np_array, rep_arr_np, axis)
        assert np.array_equal(res_num3, res_np3)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
